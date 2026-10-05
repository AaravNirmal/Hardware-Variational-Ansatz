"""Minimal VQE engine: ideal statevector training, optional noisy evaluation.

Two modes
---------
* default           train on the ideal statevector (fast), then push the optimised
                    circuit through the noise model to get the noisy state.
* noisy_training    train directly on the noisy density-matrix energy (COBYLA,
                    slow; only sensible for small n / few parameters).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from src.noise import BASIS_GATES, NoiseConfig, build_noise_model


@dataclass
class VQEResult:
    params: np.ndarray
    energy: float  # ideal energy of the optimised circuit
    noisy_energy: float  # exact (shot-free) energy under the noise model
    n_evals: int
    converged: bool


class VQEEngine:
    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        ansatz: QuantumCircuit,
        noise: NoiseConfig | None = None,
        noisy_training: bool = False,
        seed: int | None = None,
    ):
        self.H = hamiltonian
        self.ansatz = ansatz
        self.noisy_training = noisy_training
        self.rng = np.random.default_rng(seed)
        self._params = list(ansatz.parameters)
        self.n_params = len(self._params)
        self._n_evals = 0

        self.noise_model = build_noise_model(noise)
        if self.noise_model is not None:
            self._backend = AerSimulator(method="density_matrix", noise_model=self.noise_model)
            self._tqc = transpile(ansatz, basis_gates=BASIS_GATES, optimization_level=0)
        elif noisy_training:
            raise ValueError("noisy_training=True requires a non-trivial NoiseConfig.")

    # -- states ------------------------------------------------------------ #
    def _bind(self, circuit: QuantumCircuit, x) -> QuantumCircuit:
        return circuit.assign_parameters(dict(zip(self._params, np.asarray(x, dtype=float))))

    def ideal_state(self, x) -> Statevector:
        return Statevector(self._bind(self.ansatz, x))

    def noisy_state(self, x) -> DensityMatrix:
        if self.noise_model is None:
            return DensityMatrix(self.ideal_state(x))
        qc = self._bind(self._tqc, x)
        qc.save_density_matrix()
        result = self._backend.run(qc).result()
        return DensityMatrix(result.data(0)["density_matrix"])

    # -- energies ---------------------------------------------------------- #
    def ideal_energy(self, x) -> float:
        self._n_evals += 1
        return float(np.real(self.ideal_state(x).expectation_value(self.H)))

    def noisy_energy(self, x) -> float:
        self._n_evals += 1
        return float(np.real(self.noisy_state(x).expectation_value(self.H)))

    # -- optimisation ------------------------------------------------------ #
    def run(
        self,
        x0: np.ndarray | None = None,
        maxiter: int = 300,
        restarts: int = 1,
        method: str | None = None,
    ) -> VQEResult:
        objective = self.noisy_energy if self.noisy_training else self.ideal_energy
        method = method or ("COBYLA" if self.noisy_training else "L-BFGS-B")
        self._n_evals = 0

        best_x, best_f, best_ok = None, np.inf, False
        for r in range(max(1, restarts)):
            start = x0 if (r == 0 and x0 is not None) else self.rng.normal(0, 0.1, self.n_params)
            res = minimize(objective, start, method=method, options={"maxiter": maxiter})
            if res.fun < best_f:
                best_x, best_f, best_ok = res.x, float(res.fun), bool(res.success)

        n_evals = self._n_evals
        return VQEResult(
            params=best_x,
            energy=self.ideal_energy(best_x),
            noisy_energy=self.noisy_energy(best_x),
            n_evals=n_evals,
            converged=best_ok,
        )


def sampled_expectation(
    state, op: SparsePauliOp, shots: int | None = None, rng: np.random.Generator | None = None
) -> float:
    """<op> in ``state``. With ``shots`` set, each Pauli term gets independent
    Gaussian shot noise with variance (1 - <P>^2) / shots."""
    rng = rng or np.random.default_rng()
    total = 0.0
    for pauli, coeff in zip(op.paulis, op.coeffs):
        e = float(np.real(state.expectation_value(pauli)))
        if shots and np.any(pauli.x | pauli.z):  # identity term has no shot noise
            e = float(np.clip(e + rng.normal(0.0, np.sqrt(max(1 - e * e, 0.0) / shots)), -1, 1))
        total += float(np.real(coeff)) * e
    return total
