import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from src.ansatz import build_ansatz, build_hea, build_hva, candidate_ansatze
from src.hamiltonian import build_hamiltonian, build_observables
from src.solvers import VQEEngine, solve_exact


def _zero_state(qc):
    return Statevector(qc.assign_parameters(np.zeros(qc.num_parameters)))


def test_hea_param_count_and_zero_state():
    qc = build_hea(n=4, depth=2)
    assert qc.num_qubits == 4
    assert qc.num_parameters == 4 * 2 * 3
    mz = build_observables(4)["mz"]
    assert _zero_state(qc).expectation_value(mz).real == pytest.approx(1.0)


@pytest.mark.parametrize("model,per_layer", [("tfim", 2), ("xxz", 4), ("xy_chain", 3)])
def test_hva_param_count(model, per_layer):
    assert build_hva(model, 4, depth=3).num_parameters == 3 * per_layer


def test_hva_tfim_starts_in_plus_state():
    mx = build_observables(4)["mx"]
    assert _zero_state(build_hva("tfim", 4, 2)).expectation_value(mx).real == pytest.approx(1.0)


def test_hva_xxz_starts_in_singlet_dimers():
    # two singlets, each -3 J for the Heisenberg point; the odd bond contributes 0
    H = build_hamiltonian("xxz", 4, J=1.0, delta=1.0)
    assert _zero_state(build_hva("xxz", 4, 2)).expectation_value(H).real == pytest.approx(-6.0)


def test_unknown_ansatz_and_model():
    with pytest.raises(ValueError):
        build_ansatz("nope", "tfim", 4, 2)
    with pytest.raises(ValueError):
        build_hva("nope", 4, 2)


@pytest.mark.parametrize(
    "model,params,kind,depth,n",
    [
        ("tfim", {"J": 1.0, "h": 1.0}, "hva", 4, 4),
        ("xxz", {"J": 1.0, "delta": 0.5}, "hva", 3, 4),
        ("tfim", {"J": 1.0, "h": 0.7}, "hea", 2, 3),  # HEA: finite-diff gradients, keep small
        ("xy_chain", {"Jx": 1.0, "Jy": 0.5, "h": 0.8}, "hva", 3, 3),
    ],
)
def test_vqe_reaches_exact_energy(model, params, kind, depth, n):
    H = build_hamiltonian(model, n, **params)
    exact = solve_exact(H).energy
    energies = [
        VQEEngine(H, circuit, seed=1).run(restarts=3, maxiter=500).energy
        for circuit in candidate_ansatze(kind, model, n, depth)
    ]
    assert min(energies) >= exact - 1e-9  # variational bound
    assert min(energies) - exact < 1e-2


def test_xy_chain_has_both_parity_candidates():
    c0, c1 = candidate_ansatze("hva", "xy_chain", 4, 2)
    assert c0.num_parameters == c1.num_parameters
    assert [i.operation.name for i in c1.data][0] == "x"
    assert len(candidate_ansatze("hea", "xy_chain", 4, 2)) == 1
