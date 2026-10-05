#!/usr/bin/env python
"""Generate a (noisy) ground-state dataset CSV for symbolic regression.

Examples
--------
  python scripts/generate_sr_data.py --model tfim --n-qubits 4 \
      --grid J=1.0 h=0.1:2.0:20 --out data/tfim_n4.csv

  python scripts/generate_sr_data.py --model xxz --n-qubits 4 --depth 4 \
      --grid J=1.0 delta=-0.5:2.0:12 --p1 0.001 --p2 0.01 --shots 4000 \
      --out data/xxz_n4_noisy.csv

Grid axis syntax:  name=start:stop:num | name=value | name=v1,v2,v3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import generate_dataset, make_grid, parse_axis, save_csv  # noqa: E402
from src.hamiltonian import MODELS, model_defaults  # noqa: E402
from src.noise import NoiseConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELS), default="tfim")
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--periodic", action="store_true", help="periodic boundary conditions")
    p.add_argument("--grid", nargs="*", default=[], help="parameter axes (see above)")
    p.add_argument("--ansatz", choices=["hva", "hea"], default="hva")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--p1", type=float, default=0.0, help="1-qubit depolarizing")
    p.add_argument("--p2", type=float, default=0.0, help="2-qubit depolarizing")
    p.add_argument("--t1", type=float, default=None, help="T1 (same units as gate times)")
    p.add_argument("--t2", type=float, default=None, help="T2 (<= 2*T1)")
    p.add_argument("--shots", type=int, default=None, help="finite-shot noise on noisy columns")
    p.add_argument("--noisy-training", action="store_true", help="train VQE on the noisy energy (slow)")
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/sr_dataset.csv")
    args = p.parse_args()

    axes = dict(parse_axis(s) for s in args.grid)
    unknown = set(axes) - set(model_defaults(args.model))
    if unknown:
        p.error(f"{sorted(unknown)} are not parameters of '{args.model}': "
                f"{sorted(model_defaults(args.model))}")
    grid = make_grid(axes) if axes else [{}]

    df = generate_dataset(
        args.model,
        args.n_qubits,
        grid,
        ansatz=args.ansatz,
        depth=args.depth,
        periodic=args.periodic,
        noise=NoiseConfig(p1=args.p1, p2=args.p2, t1=args.t1, t2=args.t2),
        shots=args.shots,
        noisy_training=args.noisy_training,
        restarts=args.restarts,
        maxiter=args.maxiter,
        warm_start=not args.no_warm_start,
        seed=args.seed,
    )
    out = save_csv(df, args.out)
    bad = int((~df["converged"]).sum())
    print(f"\nWrote {len(df)} rows x {df.shape[1]} cols -> {out}")
    print(f"max |E_vqe - E_exact| = {df['vqe_error'].abs().max():.2e}"
          + (f"   ({bad} point(s) did not report convergence)" if bad else ""))


if __name__ == "__main__":
    main()
