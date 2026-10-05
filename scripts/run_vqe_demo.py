#!/usr/bin/env python
"""Single-point VQE demo: exact vs ideal VQE vs noisy VQE on the TFIM."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ansatz import build_ansatz  # noqa: E402
from src.hamiltonian import build_hamiltonian, build_observables  # noqa: E402
from src.noise import NoiseConfig  # noqa: E402
from src.solvers import VQEEngine, expectation, solve_exact  # noqa: E402

N, DEPTH = 4, 3
PARAMS = {"J": 1.0, "h": 1.0}
NOISE = NoiseConfig(p1=0.001, p2=0.01)


def main() -> None:
    H = build_hamiltonian("tfim", N, **PARAMS)
    exact = solve_exact(H)
    circuit = build_ansatz("hva", "tfim", N, DEPTH)

    engine = VQEEngine(H, circuit, noise=NOISE, seed=0)
    res = engine.run()
    ideal, noisy = engine.ideal_state(res.params), engine.noisy_state(res.params)

    print(f"TFIM  n={N}  {PARAMS}  HVA depth={DEPTH} ({engine.n_params} params)")
    print(f"noise: p1={NOISE.p1}, p2={NOISE.p2}\n")
    print(f"{'':8s}{'exact':>12s}{'VQE ideal':>12s}{'VQE noisy':>12s}")
    print(f"{'E':8s}{exact.energy:12.6f}{res.energy:12.6f}{res.noisy_energy:12.6f}")
    for name, op in build_observables(N).items():
        print(f"{name:8s}{expectation(exact.state, op):12.6f}"
              f"{expectation(ideal, op):12.6f}{expectation(noisy, op):12.6f}")
    print(f"\nconverged={res.converged}  evaluations={res.n_evals}")


if __name__ == "__main__":
    main()
