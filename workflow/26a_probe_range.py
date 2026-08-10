"""Diagnostic: is log2 abundance 33 a real bound, and how far outside does Step 4 go?

The Step-4 verification flagged a predicted maximum of 33.04 against the plan's
[9, 33] window. Before deciding what to do about it, measure the *observed*
dynamic range of the measured data: if the truth itself exceeds 33, the window is
a convention rather than a physical limit; if it does not, the prediction is
extrapolating and should be constrained.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_common as S4  # noqa: E402

DATA = S4.DATA


def qs(name: str, a: np.ndarray) -> None:
    f = a[np.isfinite(a)]
    print(
        f"  {name:28s} n={f.size:>12,}  min={f.min():7.3f}  "
        f"q0.01={np.quantile(f, 0.0001):7.3f}  q99.99={np.quantile(f, 0.9999):7.3f}  "
        f"max={f.max():7.3f}"
    )
    for thr in (32.0, 32.75, 33.0):
        print(f"      cells > {thr:5.2f}: {int((f > thr).sum()):,} "
              f"({100 * (f > thr).mean():.6f}%)")


def main() -> None:
    print("=== observed log2 abundance (measured truth) ===")
    tv = pd.read_parquet(DATA / "log2_train_val.parquet")
    cols = [c for c in tv.columns if c != "sample_ID"]
    qs("train_val truth", tv[cols].to_numpy("float32"))
    del tv
    te = pd.read_parquet(DATA / "log2_test.parquet")
    cols = [c for c in te.columns if c != "sample_ID"]
    qs("test truth", te[cols].to_numpy("float32"))
    del te

    print("\n=== Step 3 submitted predictions ===")
    p3 = pd.read_parquet(DATA / "gbdt_test_predictions.parquet")
    cols = [c for c in p3.columns if c != "sample_ID"]
    qs("step3 predictions", p3[cols].to_numpy("float32"))
    del p3

    p4 = DATA / "step4_test_predictions.parquet"
    if p4.exists():
        print("\n=== Step 4 predictions (before any clipping) ===")
        d4 = pd.read_parquet(p4)
        cols = [c for c in d4.columns if c != "sample_ID"]
        a = d4[cols].to_numpy("float32")
        qs("step4 predictions", a)
        n_over = int((a > 33.0).sum())
        print(f"\n  cells strictly above 33.0: {n_over:,} of {a.size:,} "
              f"({100 * n_over / a.size:.8f}%)")
        if n_over:
            r, c = np.unravel_index(np.argsort(-a, axis=None)[:8], a.shape)
            print("  top 8 predicted values:")
            for i, j in zip(r, c):
                print(f"    sample={d4['sample_ID'].iloc[i]} protein={cols[j]} "
                      f"value={a[i, j]:.4f}")


if __name__ == "__main__":
    main()
