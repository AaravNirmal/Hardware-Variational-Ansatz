"""Hamiltonian variational ansatz (HVA): layers mirror the terms of the model.

All parameters are shared across sites within a layer, so the parameter count
is tiny (and independent of n).

    model      initial state     params / layer
    tfim       |+>^n             2  (zz, x)
    xxz        singlet dimers    4  (xy, zz) x (odd bonds, even bonds)
    xy_chain   |0>^n or |10..0>  3  (xx, yy, z)

The XXZ HVA starts from singlets on the even bonds (0,1), (2,3), ... and conserves
total Sz, so it only reaches the Sz=0 sector (use even n and h = 0).

The xy_chain HVA conserves parity prod(Z), and the ground-state parity can flip as
the parameters change (finite-size Kitaev-chain physics). ``odd_parity`` selects the
starting sector; ``candidate_ansatze`` in ``src.ansatz`` returns both.
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from src.hamiltonian import bonds


def build_hva(
    model: str, n: int, depth: int, periodic: bool = False, odd_parity: bool = False
) -> QuantumCircuit:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    bs = bonds(n, periodic)
    qc = QuantumCircuit(n, name=f"HVA-{model}")

    if model == "tfim":
        th = ParameterVector("theta", 2 * depth)
        for q in range(n):
            qc.h(q)
        for layer in range(depth):
            for i, j in bs:
                qc.rzz(th[2 * layer], i, j)
            for q in range(n):
                qc.rx(th[2 * layer + 1], q)

    elif model == "xxz":
        th = ParameterVector("theta", 4 * depth)
        for i, j in bs[0::2]:  # singlet (|01> - |10>)/sqrt2 on each even bond
            qc.h(i)
            qc.cx(i, j)
            qc.x(j)
            qc.z(i)
        for layer in range(depth):
            # odd bonds first: the dimer state is an eigenstate of the even bonds
            for s, group in enumerate((bs[1::2], bs[0::2])):
                a = th[4 * layer + 2 * s]
                b = th[4 * layer + 2 * s + 1]
                for i, j in group:
                    qc.rxx(a, i, j)
                    qc.ryy(a, i, j)
                    qc.rzz(b, i, j)

    elif model == "xy_chain":
        th = ParameterVector("theta", 3 * depth)
        if odd_parity:
            qc.x(0)
        for layer in range(depth):
            for i, j in bs:
                qc.rxx(th[3 * layer], i, j)
                qc.ryy(th[3 * layer + 1], i, j)
            for q in range(n):
                qc.rz(th[3 * layer + 2], q)

    else:
        raise ValueError(f"No HVA defined for model '{model}'. Use ansatz='hea'.")
    return qc
