import numpy as np
import pandas as pd
import pytest

from src.dataset import generate_dataset, make_grid, parse_axis, save_csv
from src.noise import NoiseConfig


def test_parse_axis():
    assert parse_axis("J=1.0") == ("J", [1.0])
    assert parse_axis("h=0:1:3") == ("h", [0.0, 0.5, 1.0])
    assert parse_axis("delta=0.5, 1, 2") == ("delta", [0.5, 1.0, 2.0])
    with pytest.raises(ValueError):
        parse_axis("oops")


def test_make_grid():
    g = make_grid({"J": [1.0, 2.0], "h": [0.1, 0.2, 0.3]})
    assert len(g) == 6 and g[0] == {"J": 1.0, "h": 0.1} and g[-1] == {"J": 2.0, "h": 0.3}


def test_noiseless_dataset():
    df = generate_dataset(
        "tfim", 3, make_grid({"h": [0.5, 1.5]}), ansatz="hva", depth=3, verbose=False
    )
    assert len(df) == 2
    for col in ["J", "h", "E_exact", "E_vqe", "E_noisy", "mz_exact", "zz_noisy", "vqe_error"]:
        assert col in df.columns
    assert (df["vqe_error"] > -1e-9).all() and (df["vqe_error"] < 1e-2).all()
    assert np.allclose(df["E_vqe"], df["E_noisy"])  # no noise -> identical


def test_noise_raises_energy_and_csv_roundtrip(tmp_path):
    df = generate_dataset(
        "tfim", 3, [{"J": 1.0, "h": 1.0}], depth=3,
        noise=NoiseConfig(p1=0.01, p2=0.05), verbose=False,
    )
    assert df.loc[0, "E_noisy"] > df.loc[0, "E_vqe"]
    path = save_csv(df, tmp_path / "sub" / "out.csv")
    back = pd.read_csv(path)
    assert list(back.columns) == list(df.columns)
    assert back.loc[0, "E_exact"] == pytest.approx(df.loc[0, "E_exact"])


def test_shot_noise_changes_noisy_columns_only():
    pt = [{"J": 1.0, "h": 1.0}]
    a = generate_dataset("tfim", 3, pt, depth=3, shots=200, seed=1, verbose=False)
    b = generate_dataset("tfim", 3, pt, depth=3, shots=None, seed=1, verbose=False)
    assert a.loc[0, "E_vqe"] == pytest.approx(b.loc[0, "E_vqe"])
    assert a.loc[0, "E_noisy"] != pytest.approx(b.loc[0, "E_noisy"], abs=1e-9)


def test_xy_chain_dataset_picks_best_parity():
    df = generate_dataset(
        "xy_chain", 3, [{"Jx": 1.0, "Jy": 0.5, "h": 0.8}], depth=3, verbose=False
    )
    assert df.loc[0, "vqe_error"] < 1e-2
