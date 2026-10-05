"""Hardware-efficient ansatz (HEA): layers of single-qubit rotations + CX ladder."""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from src.hamiltonian import bonds


def build_hea(
    n: int,
    depth: int,
    rotations: tuple[str, ...] = ("ry", "rz"),
    periodic: bool = False,
) -> QuantumCircuit:
    """``depth`` entangling layers, each preceded by one rotation layer, plus a
    final rotation layer. Parameter count: ``n * len(rotations) * (depth + 1)``.
    Zero parameters give |0...0>.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    theta = ParameterVector("theta", n * len(rotations) * (depth + 1))
    qc = QuantumCircuit(n, name="HEA")
    k = 0
    for layer in range(depth + 1):
        for q in range(n):
            for rot in rotations:
                getattr(qc, rot)(theta[k], q)
                k += 1
        if layer < depth:
            for i, j in bonds(n, periodic):
                qc.cx(i, j)
            qc.barrier()
    return qc
