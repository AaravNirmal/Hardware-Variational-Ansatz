from qiskit import QuantumCircuit

from .hea import build_hea
from .hva import build_hva


def build_ansatz(
    kind: str, model: str, n: int, depth: int, periodic: bool = False
) -> QuantumCircuit:
    """Factory: ``kind`` is 'hea' (model-agnostic) or 'hva' (model-specific)."""
    if kind == "hea":
        return build_hea(n, depth, periodic=periodic)
    if kind == "hva":
        return build_hva(model, n, depth, periodic=periodic)
    raise ValueError(f"Unknown ansatz '{kind}'. Use 'hea' or 'hva'.")


def candidate_ansatze(
    kind: str, model: str, n: int, depth: int, periodic: bool = False
) -> list[QuantumCircuit]:
    """Circuits to try per Hamiltonian point (the lowest-energy result is kept).

    Normally one circuit. For the parity-conserving xy_chain HVA it is one circuit
    per parity sector, because the true ground state may sit in either.
    """
    if kind == "hva" and model == "xy_chain":
        return [build_hva(model, n, depth, periodic, odd_parity=odd) for odd in (False, True)]
    return [build_ansatz(kind, model, n, depth, periodic)]


__all__ = ["build_hea", "build_hva", "build_ansatz", "candidate_ansatze"]
