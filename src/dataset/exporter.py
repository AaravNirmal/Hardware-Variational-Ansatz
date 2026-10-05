"""Parameter sweeps -> pandas DataFrame -> CSV for the Auto-SR tool.

One CSV row per Hamiltonian-parameter point. Column groups:

  meta         model, n_qubits, periodic, ansatz, depth, p1, p2, t1, t2, shots
  parameters   one column per Hamiltonian parameter (J, h, delta, ...)
  energies     E_{exact,vqe,noisy}  (total)   e_{exact,vqe,noisy}  (per site)
  observables  {mx,my,mz,xx,yy,zz}_{exact,vqe,noisy}
  diagnostics  gap, vqe_error, converged, n_evals

  *_exact  exact diagonalisation       *_vqe    optimised ansatz, noiseless
  *_noisy  same circuit through the noise model (+ finite-shot noise if shots set)
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.ansatz import candidate_ansatze
from src.hamiltonian import build_hamiltonian, build_observables, model_defaults
from src.noise import NoiseConfig
from src.solvers import VQEEngine, expectation, sampled_expectation, solve_exact


# --------------------------------------------------------------------------- #
# Grid helpers
# --------------------------------------------------------------------------- #
def parse_axis(spec: str) -> tuple[str, list[float]]:
    """'h=0.2:2.0:10' (linspace) | 'J=1.0' | 'delta=0.5,1,2' -> (name, values)."""
    name, _, rhs = spec.partition("=")
    name, rhs = name.strip(), rhs.strip()
    if not name or not rhs:
        raise ValueError(f"Bad axis spec '{spec}'. Use name=start:stop:num or name=v1,v2,...")
    if ":" in rhs:
        start, stop, num = rhs.split(":")
        values = np.linspace(float(start), float(stop), int(num))
    else:
        values = [float(v) for v in rhs.split(",")]
    return name, [float(v) for v in values]


def make_grid(axes: Mapping[str, Sequence[float]]) -> list[dict[str, float]]:
    """Cartesian product of parameter axes (last axis varies fastest)."""
    names = list(axes)
    return [dict(zip(names, combo)) for combo in itertools.product(*(axes[k] for k in names))]


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
def generate_dataset(
    model: str,
    n_qubits: int,
    grid: Sequence[Mapping[str, float]],
    *,
    ansatz: str = "hva",
    depth: int = 3,
    periodic: bool = False,
    noise: NoiseConfig | None = None,
    shots: int | None = None,
    noisy_training: bool = False,
    restarts: int = 1,
    maxiter: int = 300,
    warm_start: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    noise = noise or NoiseConfig()
    observables = build_observables(n_qubits, periodic)
    circuits = candidate_ansatze(ansatz, model, n_qubits, depth, periodic)
    defaults = model_defaults(model)

    rows, prev = [], [None] * len(circuits)
    for idx, point in enumerate(grid):
        params = {**defaults, **point}
        H = build_hamiltonian(model, n_qubits, periodic, **params)

        exact = solve_exact(H)
        best = None  # (engine, result) of the lowest-energy candidate circuit
        for c, circuit in enumerate(circuits):
            eng = VQEEngine(H, circuit, noise=noise, noisy_training=noisy_training, seed=seed + idx)
            r = eng.run(x0=prev[c] if warm_start else None, maxiter=maxiter, restarts=restarts)
            prev[c] = r.params
            if best is None or r.energy < best[1].energy:
                best = (eng, r)
        engine, res = best

        vqe_state = engine.ideal_state(res.params)
        noisy_state = engine.noisy_state(res.params)
        rng = np.random.default_rng(seed + 10_000 + idx)

        row: dict = {
            "model": model,
            "n_qubits": n_qubits,
            "periodic": periodic,
            "ansatz": ansatz,
            "depth": depth,
            "p1": noise.p1,
            "p2": noise.p2,
            "t1": np.nan if noise.t1 is None else noise.t1,
            "t2": np.nan if noise.t2 is None else noise.t2,
            "shots": np.nan if shots is None else shots,
            **params,
        }
        E_exact = exact.energy
        E_vqe = res.energy
        E_noisy = sampled_expectation(noisy_state, H, shots, rng)
        for tag, E in (("exact", E_exact), ("vqe", E_vqe), ("noisy", E_noisy)):
            row[f"E_{tag}"] = E
            row[f"e_{tag}"] = E / n_qubits
        for name, op in observables.items():
            row[f"{name}_exact"] = expectation(exact.state, op)
            row[f"{name}_vqe"] = expectation(vqe_state, op)
            row[f"{name}_noisy"] = sampled_expectation(noisy_state, op, shots, rng)
        row.update(
            gap=exact.gap,
            vqe_error=E_vqe - E_exact,
            converged=res.converged,
            n_evals=res.n_evals,
        )
        rows.append(row)

        if verbose:
            print(
                f"[{idx + 1}/{len(grid)}] "
                + " ".join(f"{k}={v:.3g}" for k, v in point.items())
                + f"  E_exact={E_exact:.5f}  E_vqe={E_vqe:.5f}  E_noisy={E_noisy:.5f}"
            )
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
