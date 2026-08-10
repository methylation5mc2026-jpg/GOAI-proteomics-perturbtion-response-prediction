"""Bounded probe of the Step-3 modelling inputs.

Reports treated-row split counts, metadata cardinalities on the columns the
feature builder will consume, and the finite-cell fraction of Y / Delta -- the
numbers that set the size of the long-format design matrix.  Read-only.
"""

from __future__ import annotations

from repo_paths import WORKFLOW_DIR

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(WORKFLOW_DIR))

from common import BIO_COLS, CHEM_COL, CTX_COLS, DATA, ID_COL  # noqa: E402


def main() -> None:
    for label, path in [("train_val", DATA / "meta_train_val_annotated.csv"),
                        ("test", DATA / "meta_test_annotated.csv")]:
        meta = pd.read_csv(path, dtype=str)
        print(f"\n=== {label}: {meta.shape[0]} rows x {meta.shape[1]} cols ===")
        print("  split_final x sample_role:")
        print(pd.crosstab(meta["split_final"], meta["sample_role"]).to_string())
        for c in BIO_COLS + CTX_COLS + ["pert_id", "pert_time_unit"]:
            u = meta[c].astype(str)
            print(f"  {c:32s} n_unique={u.nunique():5d} "
                  f"n_na={int(meta[c].isna().sum()):5d} "
                  f"examples={sorted(u.unique())[:4]}")

    # pert_time / Temperature must be numerically parseable for the numeric
    # encoding the plan calls for.
    meta = pd.read_csv(DATA / "meta_train_val_annotated.csv", dtype=str)
    for c in ["pert_time", "Temperature"]:
        v = pd.to_numeric(meta[c], errors="coerce")
        print(f"\n  {c}: numeric-parse failures={int(v.isna().sum())}, "
              f"values={sorted(v.dropna().unique().tolist())}")

    print("\n  protein_well pattern (row letter / column number decomposition):")
    w = meta["protein_well"].astype(str)
    print(f"    examples={sorted(w.unique())[:8]} n_unique={w.nunique()}")
    row = w.str.extract(r"^([A-Za-z]+)")[0]
    col = pd.to_numeric(w.str.extract(r"(\d+)$")[0], errors="coerce")
    print(f"    row letters={sorted(row.dropna().unique())}")
    print(f"    col numbers={sorted(col.dropna().unique().astype(int).tolist())}")
    print(f"    unparsed wells={int(row.isna().sum() + col.isna().sum())}")

    # Finite-cell fractions drive the long-format row budget.
    import pyarrow.parquet as pq
    from common import WORKFLOW
    for nm in ["processed_delta_matrix.parquet", "processed_train_val_proteome.parquet",
               "processed_delta_matrix_test.parquet", "processed_test_proteome.parquet"]:
        md = pq.read_metadata(WORKFLOW / nm)
        print(f"  {nm}: {md.num_rows} rows x {md.num_columns} cols")

    d = pd.read_parquet(WORKFLOW / "processed_delta_matrix.parquet")
    prot = [c for c in d.columns if c not in (ID_COL, "match_level", "split_final", CHEM_COL)]
    D = d[prot].to_numpy(dtype="float32")
    print(f"\n  Delta train_val: {D.shape}, finite={100 * np.isfinite(D).mean():.2f}% "
          f"-> {int(np.isfinite(D).sum()):,} finite cells")
    sp = d["split_final"].to_numpy()
    for s in np.unique(sp):
        m = sp == s
        print(f"    {s:18s} rows={int(m.sum()):5d} finite_cells={int(np.isfinite(D[m]).sum()):>11,}")


if __name__ == "__main__":
    main()
