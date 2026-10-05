"""Spin-chain Hamiltonians and observables as Qiskit ``SparsePauliOp``s.

Conventions (open boundary conditions unless ``periodic=True``)::

    tfim      H = -J  sum Z_i Z_{i+1}  -  h sum X_i
    xxz       H =  J  sum (X_i X_{i+1} + Y_i Y_{i+1} + delta Z_i Z_{i+1})  +  h sum Z_i
    xy_chain  H = -Jx sum X_i X_{i+1}  -  Jy sum Y_i Y_{i+1}  -  h sum Z_i

``xy_chain`` is the spin form of a Kitaev (p-wave) chain: under a Jordan-Wigner
transform, Jx+Jy plays the role of hopping and Jx-Jy the pairing amplitude
(up to sign conventions), and h plays the role of the chemical potential.

To add a model: write a builder ``f(n, periodic=False, **params)`` returning a
``SparsePauliOp`` and register it in ``MODELS``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from qiskit.quantum_info import SparsePauliOp


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def bonds(n: int, periodic: bool = False) -> list[tuple[int, int]]:
    """Nearest-neighbour bonds of a 1D chain of ``n`` sites."""
    if n < 2:
        raise ValueError("Need at least 2 qubits.")
    out = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        out.append((n - 1, 0))
    return out


def _assemble(n: int, terms: list[tuple[str, list[int], float]]) -> SparsePauliOp:
    if not terms:
        return SparsePauliOp("I" * n, coeffs=[0.0])
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def tfim(n: int, periodic: bool = False, J: float = 1.0, h: float = 1.0) -> SparsePauliOp:
    terms = [("ZZ", [i, j], -J) for i, j in bonds(n, periodic)]
    terms += [("X", [i], -h) for i in range(n)]
    return _assemble(n, terms)


def xxz(
    n: int, periodic: bool = False, J: float = 1.0, delta: float = 1.0, h: float = 0.0
) -> SparsePauliOp:
    terms: list[tuple[str, list[int], float]] = []
    for i, j in bonds(n, periodic):
        terms += [("XX", [i, j], J), ("YY", [i, j], J), ("ZZ", [i, j], J * delta)]
    terms += [("Z", [i], h) for i in range(n)]
    return _assemble(n, terms)


def xy_chain(
    n: int, periodic: bool = False, Jx: float = 1.0, Jy: float = 1.0, h: float = 0.0
) -> SparsePauliOp:
    terms: list[tuple[str, list[int], float]] = []
    for i, j in bonds(n, periodic):
        terms += [("XX", [i, j], -Jx), ("YY", [i, j], -Jy)]
    terms += [("Z", [i], -h) for i in range(n)]
    return _assemble(n, terms)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[..., SparsePauliOp]
    defaults: dict
    description: str


MODELS: dict[str, ModelSpec] = {
    "tfim": ModelSpec(tfim, {"J": 1.0, "h": 1.0}, "Transverse-field Ising chain"),
    "xxz": ModelSpec(
        xxz, {"J": 1.0, "delta": 1.0, "h": 0.0}, "Anisotropic Heisenberg (XXZ) chain"
    ),
    "xy_chain": ModelSpec(
        xy_chain,
        {"Jx": 1.0, "Jy": 1.0, "h": 0.0},
        "Anisotropic XY chain in a field (spin form of the Kitaev chain)",
    ),
}


def _spec(model: str) -> ModelSpec:
    try:
        return MODELS[model]
    except KeyError:
        raise ValueError(f"Unknown model '{model}'. Available: {sorted(MODELS)}") from None


def model_defaults(model: str) -> dict:
    return dict(_spec(model).defaults)


def build_hamiltonian(model: str, n: int, periodic: bool = False, **params) -> SparsePauliOp:
    """Build ``model`` on ``n`` qubits; unspecified params take their defaults."""
    spec = _spec(model)
    unknown = set(params) - set(spec.defaults)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {sorted(unknown)} for '{model}'. "
            f"Valid: {sorted(spec.defaults)}"
        )
    return spec.builder(n, periodic=periodic, **{**spec.defaults, **params})


# --------------------------------------------------------------------------- #
# Observables
# --------------------------------------------------------------------------- #
def build_observables(n: int, periodic: bool = False) -> dict[str, SparsePauliOp]:
    """Site-averaged magnetisations (mx, my, mz) and bond-averaged
    nearest-neighbour correlators (xx, yy, zz)."""
    bs = bonds(n, periodic)
    obs: dict[str, SparsePauliOp] = {}
    for p in "XYZ":
        obs[f"m{p.lower()}"] = _assemble(n, [(p, [i], 1.0 / n) for i in range(n)])
        obs[p.lower() * 2] = _assemble(n, [(p + p, [i, j], 1.0 / len(bs)) for i, j in bs])
    return obs
