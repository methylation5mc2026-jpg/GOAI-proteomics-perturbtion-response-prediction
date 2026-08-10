"""Probe: enumerate chemical perturbations and split structure for Step 4.

Writes a JSON profile to results/step4_chemical_probe.json so the RDKit
feature builder knows exactly which compound names must be resolved.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
from pathlib import Path

import pandas as pd

SESSION = REPO_ROOT
DATA = SESSION / "data"
RESULTS = SESSION / "results"


def main() -> None:
    tr = pd.read_csv(DATA / "meta_train_val_annotated.csv")
    te = pd.read_csv(DATA / "meta_test_annotated.csv")
    print(f"train_val {tr.shape}  test {te.shape}")

    for name, df in (("train_val", tr), ("test", te)):
        print(f"--- {name}: split_final ---")
        print(df["split_final"].value_counts().to_string())
        print(f"--- {name}: sample_role ---")
        print(df["sample_role"].value_counts().to_string())

    col = "perturbation_no_concentration"
    a = set(tr[col].dropna().astype(str))
    b = set(te[col].dropna().astype(str))
    print(f"\nn_chem train={len(a)} test={len(b)} union={len(a | b)} test_only={len(b - a)}")

    # chemicals appearing on the novel-chemical validation split
    val_chem = set(
        tr.loc[tr["split_final"].astype(str) == "val_chem_only", col].dropna().astype(str)
    )
    print(f"val_chem_only chemicals: {len(val_chem)}")

    profile = {
        "n_rows_train_val": int(tr.shape[0]),
        "n_rows_test": int(te.shape[0]),
        "split_counts_train_val": {
            str(k): int(v) for k, v in tr["split_final"].value_counts().items()
        },
        "split_counts_test": {
            str(k): int(v) for k, v in te["split_final"].value_counts().items()
        },
        "n_chem_train": len(a),
        "n_chem_test": len(b),
        "n_chem_union": len(a | b),
        "chem_test_only": sorted(b - a),
        "chem_val_chem_only": sorted(val_chem),
        "chem_all": sorted(a | b),
        "strains_train": sorted(set(tr["Strains"].dropna().astype(str))),
        "strains_test_only": sorted(
            set(te["Strains"].dropna().astype(str)) - set(tr["Strains"].dropna().astype(str))
        ),
    }
    out = RESULTS / "step4_chemical_probe.json"
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("\nALL CHEMICAL NAMES:")
    for c in profile["chem_all"]:
        print(f"  {c}")


if __name__ == "__main__":
    main()
