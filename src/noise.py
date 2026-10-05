"""Noise configuration -> Qiskit Aer NoiseModel."""
from __future__ import annotations

from dataclasses import dataclass

from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

# Gate set the noisy simulator runs on. Circuits are transpiled to this basis.
BASIS_1Q = ["rx", "ry", "rz", "h", "x"]
BASIS_2Q = ["cx"]
BASIS_GATES = BASIS_1Q + BASIS_2Q


@dataclass(frozen=True)
class NoiseConfig:
    """Gate noise parameters.

    p1, p2        depolarizing parameter after every 1-/2-qubit gate
    t1, t2        relaxation / dephasing times (same unit as gate times); None = off
    gate_time_1q  duration of a 1-qubit gate
    gate_time_2q  duration of a 2-qubit gate
    """

    p1: float = 0.0
    p2: float = 0.0
    t1: float | None = None
    t2: float | None = None
    gate_time_1q: float = 0.05
    gate_time_2q: float = 0.30

    @property
    def is_noiseless(self) -> bool:
        return self.p1 == 0.0 and self.p2 == 0.0 and self.t1 is None


def build_noise_model(cfg: NoiseConfig | None) -> NoiseModel | None:
    """Return an Aer ``NoiseModel`` (or ``None`` if the config is noiseless)."""
    if cfg is None or cfg.is_noiseless:
        return None

    err1 = err2 = None
    if cfg.t1 is not None:
        t2 = cfg.t2 if cfg.t2 is not None else cfg.t1
        t2 = min(t2, 2 * cfg.t1)  # physical constraint T2 <= 2 T1
        err1 = thermal_relaxation_error(cfg.t1, t2, cfg.gate_time_1q)
        r2 = thermal_relaxation_error(cfg.t1, t2, cfg.gate_time_2q)
        err2 = r2.tensor(r2)
    if cfg.p1 > 0:
        dep = depolarizing_error(cfg.p1, 1)
        err1 = dep if err1 is None else err1.compose(dep)
    if cfg.p2 > 0:
        dep = depolarizing_error(cfg.p2, 2)
        err2 = dep if err2 is None else err2.compose(dep)

    model = NoiseModel(basis_gates=BASIS_GATES)
    if err1 is not None:
        model.add_all_qubit_quantum_error(err1, BASIS_1Q)
    if err2 is not None:
        model.add_all_qubit_quantum_error(err2, BASIS_2Q)
    return model
