import numpy as np
import pytest

from src.hamiltonian import (
    MODELS,
    bonds,
    build_hamiltonian,
    build_observables,
    model_defaults,
)
from src.solvers import solve_exact


def test_bonds():
    assert bonds(4) == [(0, 1), (1, 2), (2, 3)]
    assert bonds(4, periodic=True)[-1] == (3, 0)
    assert bonds(2, periodic=True) == [(0, 1)]  # no double-counted bond
    with pytest.raises(ValueError):
        bonds(1)


def test_tfim_term_counts():
    assert len(build_hamiltonian("tfim", 3)) == 5  # 2 ZZ + 3 X
    assert len(build_hamiltonian("tfim", 3, periodic=True)) == 6


def test_tfim_limits():
    assert solve_exact(build_hamiltonian("tfim", 4, J=0.0, h=1.0)).energy == pytest.approx(-4.0)
    assert solve_exact(build_hamiltonian("tfim", 4, J=1.0, h=0.0)).energy == pytest.approx(-3.0)


def test_tfim_two_site_analytic():
    # H = -J ZZ - h (X0 + X1): E0 = -sqrt(J^2 + 4 h^2)
    J, h = 0.7, 1.3
    e = solve_exact(build_hamiltonian("tfim", 2, J=J, h=h)).energy
    assert e == pytest.approx(-np.sqrt(J**2 + 4 * h**2))


def test_xxz_heisenberg_dimer_is_singlet():
    assert solve_exact(build_hamiltonian("xxz", 2, J=1.0, delta=1.0)).energy == pytest.approx(-3.0)


def test_xy_chain_limits():
    assert solve_exact(build_hamiltonian("xy_chain", 2, Jx=1.0, Jy=0.0, h=0.0)).energy == pytest.approx(-1.0)
    assert solve_exact(build_hamiltonian("xy_chain", 2, Jx=0.0, Jy=0.0, h=1.0)).energy == pytest.approx(-2.0)


@pytest.mark.parametrize("model", sorted(MODELS))
def test_hermitian(model):
    m = build_hamiltonian(model, 3).to_matrix()
    assert np.allclose(m, m.conj().T)


def test_unknown_model_and_param():
    with pytest.raises(ValueError):
        build_hamiltonian("nope", 3)
    with pytest.raises(ValueError):
        build_hamiltonian("tfim", 3, bogus=1.0)


def test_defaults_are_copies():
    d = model_defaults("tfim")
    d["J"] = 99
    assert model_defaults("tfim")["J"] == 1.0


def test_observables():
    obs = build_observables(4)
    assert set(obs) == {"mx", "my", "mz", "xx", "yy", "zz"}
