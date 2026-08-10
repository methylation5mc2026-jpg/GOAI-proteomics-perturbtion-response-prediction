"""
Step 1d: Probe the proteome matrices' scale and encoding before any transform.

Critical question: are the values in *_proteome_raw_*.csv linear intensities or
already log2-transformed? The answer decides whether a log2 step is needed at
all. This reads a bounded row sample (not the full 277 MB file) and reports the
observed numeric range, zero/NaN encoding and per-sample totals.
"""

from __future__ import annotations

from common import PROT_TE, PROT_TV, require_input_files
from repo_paths import REPO_ROOT, RESULTS_DIR

import json
from pathlib import Path

import numpy as np
import pandas as pd

SESSION = REPO_ROOT
RESULTS = RESULTS_DIR

N_ROWS = 400  # bounded probe

np.random.seed(42)


def probe(path: Path, label: str) -> dict:
    print(f"\n--- {label}: bounded probe of first {N_ROWS} rows ---")
    df = pd.read_csv(path, nrows=N_ROWS)
    ids = df.iloc[:, 0]
    vals = df.iloc[:, 1:]
    print(f"  shape (probe): {df.shape}; id column name: {df.columns[0]!r}")
    print(f"  value dtypes unique: {sorted(set(map(str, vals.dtypes)))}")

    arr = vals.to_numpy(dtype="float64", na_value=np.nan)
    finite = arr[np.isfinite(arr)]
    n_tot = arr.size
    n_nan = int(np.isnan(arr).sum())
    n_zero = int((arr == 0).sum())
    n_neg = int((finite < 0).sum())

    q = np.percentile(finite[finite != 0], [0, 1, 25, 50, 75, 99, 100]) if finite.size else []
    print(f"  cells={n_tot}  NaN={n_nan} ({100*n_nan/n_tot:.2f}%)  "
          f"exact-zero={n_zero} ({100*n_zero/n_tot:.2f}%)  negative={n_neg}")
    print(f"  nonzero-finite percentiles [0,1,25,50,75,99,100]: "
          f"{np.round(q, 4).tolist()}")

    # Per-sample summaries on the probe
    row_sum = np.nansum(arr, axis=1)
    row_det = np.isfinite(arr).sum(axis=1)
    print(f"  per-sample detected proteins: median={np.median(row_det):.0f} "
          f"min={row_det.min()} max={row_det.max()} (of {arr.shape[1]})")
    print(f"  per-sample sum of values: median={np.median(row_sum):.3f}")

    # Scale verdict
    mx = float(np.nanmax(finite)) if finite.size else float("nan")
    already_log = mx < 64 and n_neg == 0
    print(f"  max value = {mx:.4f}  -> "
          f"{'LOOKS ALREADY log2-SCALED' if already_log else 'looks LINEAR intensity'}")

    # Duplicate protein columns?
    cols = list(vals.columns)
    dup = pd.Series(cols).duplicated().sum()
    print(f"  n protein columns={len(cols)}  duplicated names={int(dup)}")
    print(f"  first 5 sample_IDs: {ids.head(5).tolist()}")

    return {
        "label": label,
        "probe_rows": int(df.shape[0]),
        "n_protein_columns": len(cols),
        "n_duplicate_protein_columns": int(dup),
        "id_column": str(df.columns[0]),
        "pct_nan": round(100 * n_nan / n_tot, 4),
        "pct_exact_zero": round(100 * n_zero / n_tot, 4),
        "n_negative": n_neg,
        "nonzero_percentiles": {str(p): float(v) for p, v in
                                zip([0, 1, 25, 50, 75, 99, 100], q)},
        "max_value": mx,
        "verdict_already_log2": bool(already_log),
        "detected_per_sample_median": float(np.median(row_det)),
    }


def main() -> None:
    require_input_files("proteome_train_val", "proteome_test")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {}
    for path, label in ((PROT_TV, "proteome_train_val"), (PROT_TE, "proteome_test")):
        out[label] = probe(path, label)

    # Confirm protein column order is identical between the two files
    h1 = pd.read_csv(PROT_TV, nrows=0).columns.tolist()
    h2 = pd.read_csv(PROT_TE, nrows=0).columns.tolist()
    same = h1 == h2
    print(f"\nProtein column order identical between train_val and test: {same}")
    print(f"  n columns: {len(h1)} vs {len(h2)}")
    out["column_order_identical"] = bool(same)
    out["n_columns"] = len(h1)

    dest = RESULTS / "proteome_probe.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {dest}")


if __name__ == "__main__":
    main()
