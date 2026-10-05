"""Exact diagonalisation reference solver."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.sparse.linalg import eigsh

DENSE_LIMIT = 10  # use dense eigh up to this many qubits


@dataclass
class ExactResult:
    energy: float
    gap: float  # E1 - E0 (~0 for a degenerate ground space)
    state: Statevector


def solve_exact(hamiltonian: SparsePauliOp) -> ExactResult:
    n = hamiltonian.num_qubits
    if n <= DENSE_LIMIT:
        evals, evecs = np.linalg.eigh(hamiltonian.to_matrix())
        e0, e1, v0 = evals[0], evals[1], evecs[:, 0]
    else:
        evals, evecs = eigsh(hamiltonian.to_matrix(sparse=True), k=2, which="SA")
        order = np.argsort(evals)
        e0, e1, v0 = evals[order[0]], evals[order[1]], evecs[:, order[0]]
    return ExactResult(float(e0), float(e1 - e0), Statevector(np.ascontiguousarray(v0)))


def expectation(state, op: SparsePauliOp) -> float:
    """Real expectation value of ``op`` in a Statevector or DensityMatrix."""
    return float(np.real(state.expectation_value(op)))
