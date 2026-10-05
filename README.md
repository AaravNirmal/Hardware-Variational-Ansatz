# Hardware-Efficient-Ansatz

Qiskit pipeline that simulates (noisy) variational ground states of spin-chain
Hamiltonians and exports the resulting energies and observables to CSV, as input
for an independent Automated Symbolic Regression (Auto-SR) tool that tries to
recover the Hamiltonian from the data.

## Models

| name       | Hamiltonian                                                   | parameters        |
|------------|---------------------------------------------------------------|-------------------|
| `tfim`     | `-J ΣZZ - h ΣX`                                               | `J`, `h`          |
| `xxz`      | `J Σ(XX + YY + Δ ZZ) + h ΣZ`                                  | `J`, `delta`, `h` |
| `xy_chain` | `-Jx ΣXX - Jy ΣYY - h ΣZ` (spin form of the Kitaev chain)    | `Jx`, `Jy`, `h`   |

Add a model by writing a builder in `src/hamiltonian.py` and registering it in `MODELS`
(plus an HVA in `src/ansatz/hva.py`, or just use `--ansatz hea`).

## Usage

```bash
pip install -r requirements.txt
python -m pytest                      # run tests
python scripts/run_vqe_demo.py        # single-point exact vs VQE vs noisy VQE

# parameter sweep -> CSV
python scripts/generate_sr_data.py --model tfim --n-qubits 4 \
    --grid J=1.0 h=0.2:2.0:20 --p1 0.001 --p2 0.01 --shots 4000 \
    --out data/tfim_n4.csv
```

Grid axes: `name=start:stop:num` (linspace), `name=value`, or `name=v1,v2,v3`.
Axes are combined as a Cartesian product.

## Noise model

`--p1 / --p2` depolarizing error per 1-/2-qubit gate; `--t1 / --t2` thermal relaxation
(same time units as the gate times in `NoiseConfig`); `--shots` adds finite-sampling noise
to the `*_noisy` columns. By default the VQE is trained on the ideal statevector and the
optimised circuit is then run through the noise model (density-matrix simulation). Use
`--noisy-training` to optimise the noisy energy directly (slow).

## CSV columns

| group       | columns |
|-------------|---------|
| metadata    | `model, n_qubits, periodic, ansatz, depth, p1, p2, t1, t2, shots` |
| parameters  | one per Hamiltonian parameter (`J`, `h`, `delta`, ...) |
| energies    | `E_*` total and `e_*` per site, for `* ∈ {exact, vqe, noisy}` |
| observables | `{mx,my,mz,xx,yy,zz}_{exact,vqe,noisy}` (site / nearest-neighbour averages) |
| diagnostics | `gap, vqe_error, converged, n_evals` |

`*_exact` = exact diagonalisation, `*_vqe` = optimised ansatz (noiseless),
`*_noisy` = same circuit under noise. Filter on `vqe_error` before regression.

## Layout

```
src/hamiltonian.py      models, observables
src/noise.py            NoiseConfig -> Aer NoiseModel
src/ansatz/             hea.py, hva.py
src/solvers/            exact_solver.py, vqe_engine.py
src/dataset/exporter.py grid sweeps -> DataFrame -> CSV
scripts/                CLI entry points
tests/                  pytest suite
```
