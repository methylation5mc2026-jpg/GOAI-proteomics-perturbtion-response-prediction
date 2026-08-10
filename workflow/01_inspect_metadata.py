"""
Step 1a: Inspect and profile the WAYB/WAYC metadata files.

Profiles every metadata column (cardinality, level counts), checks sample_ID
uniqueness and overlap between train_val and test, and cross-tabulates the
official evaluation splits against strain_role / chemical_role. Writes a
machine-readable profile to results/metadata_profile.json.

No proteome values are loaded here - this is metadata only, so it is cheap.
"""

from __future__ import annotations

from common import load_metadata
from repo_paths import LOGS_DIR, REPO_ROOT, RESULTS_DIR

import json

import numpy as np
import pandas as pd

SESSION = REPO_ROOT
RESULTS = RESULTS_DIR
LOGS = LOGS_DIR

np.random.seed(42)


def profile_frame(df: pd.DataFrame, name: str) -> dict:
    """Profile a metadata table column-by-column.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table.
    name : str
        Label used in the returned profile.

    Returns
    -------
    dict
        Column-wise profile including dtype, missing counts and value levels
        (full level list when cardinality <= 60, else the 20 most frequent).
    """
    prof = {"name": name, "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]),
            "columns": {}}
    for col in df.columns:
        s = df[col]
        n_uniq = int(s.nunique(dropna=True))
        entry = {
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "n_unique": n_uniq,
        }
        vc = s.value_counts(dropna=False)
        if n_uniq <= 60:
            entry["levels"] = {str(k): int(v) for k, v in vc.items()}
        else:
            entry["top20_levels"] = {str(k): int(v) for k, v in vc.head(20).items()}
        prof["columns"][col] = entry
    return prof


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    print("[1/5] Reading metadata files (all columns as string to avoid coercion)...")
    meta_tv, meta_te = load_metadata()
    print(f"  train_val metadata: {meta_tv.shape}")
    print(f"  test      metadata: {meta_te.shape}")
    print(f"  columns identical: {list(meta_tv.columns) == list(meta_te.columns)}")

    print("[2/5] Checking sample_ID uniqueness / overlap...")
    ids_tv, ids_te = meta_tv["sample_ID"], meta_te["sample_ID"]
    id_checks = {
        "train_val_n": int(len(ids_tv)),
        "train_val_n_unique": int(ids_tv.nunique()),
        "train_val_duplicated": int(ids_tv.duplicated().sum()),
        "test_n": int(len(ids_te)),
        "test_n_unique": int(ids_te.nunique()),
        "test_duplicated": int(ids_te.duplicated().sum()),
        "n_overlap_tv_test": int(len(set(ids_tv) & set(ids_te))),
    }
    for k, v in id_checks.items():
        print(f"  {k}: {v}")

    print("[3/5] Profiling columns...")
    prof_tv = profile_frame(meta_tv, "metadata_train_val")
    prof_te = profile_frame(meta_te, "metadata_test")

    for prof in (prof_tv, prof_te):
        print(f"\n  --- {prof['name']} ({prof['n_rows']} rows) ---")
        for col, e in prof["columns"].items():
            lv = e.get("levels")
            desc = (f"{list(lv)[:12]}{'...' if len(lv) > 12 else ''}" if lv
                    else f"(high-cardinality, top: {list(e['top20_levels'])[:5]})")
            print(f"   {col:<32} n_unique={e['n_unique']:<6} missing={e['n_missing']:<5} {desc}")

    print("\n[4/5] Cross-tabulating splits vs strain_role / chemical_role...")
    crosstabs = {}
    for label, df in (("train_val", meta_tv), ("test", meta_te)):
        ct = pd.crosstab(df["split_final"], [df["strain_role"], df["chemical_role"]])
        print(f"\n  --- {label}: split_final x (strain_role, chemical_role) ---")
        print(ct.to_string())
        crosstabs[f"{label}_split_x_roles"] = json.loads(
            ct.reset_index().to_json(orient="records")
        )
        # split x batch
        ctb = pd.crosstab(df["split_final"], df["data_source"])
        print(f"\n  --- {label}: split_final x data_source ---")
        print(ctb.to_string())
        crosstabs[f"{label}_split_x_batch"] = json.loads(
            ctb.reset_index().to_json(orient="records")
        )

    print("\n[5/5] Strain / chemical entity visibility across train_val vs test...")
    strains_tv = set(meta_tv["Strains"].dropna())
    strains_te = set(meta_te["Strains"].dropna())
    chems_tv = set(meta_tv["perturbation_no_concentration"].dropna())
    chems_te = set(meta_te["perturbation_no_concentration"].dropna())

    # Entities visible in the *train* split only (this is what defines OOD-ness)
    tr_mask = meta_tv["split_final"] == "train"
    strains_train = set(meta_tv.loc[tr_mask, "Strains"].dropna())
    chems_train = set(meta_tv.loc[tr_mask, "perturbation_no_concentration"].dropna())

    entity = {
        "n_strains_train_val": len(strains_tv),
        "n_strains_test": len(strains_te),
        "n_strains_shared": len(strains_tv & strains_te),
        "n_strains_test_only": len(strains_te - strains_tv),
        "n_strains_in_train_split": len(strains_train),
        "n_chems_train_val": len(chems_tv),
        "n_chems_test": len(chems_te),
        "n_chems_shared": len(chems_tv & chems_te),
        "n_chems_test_only": len(chems_te - chems_tv),
        "n_chems_in_train_split": len(chems_train),
        "chems_all_sorted": sorted(chems_tv | chems_te),
        "strains_test_only_sorted": sorted(strains_te - strains_tv),
        "chems_test_only_sorted": sorted(chems_te - chems_tv),
    }
    for k, v in entity.items():
        if not k.endswith("_sorted"):
            print(f"  {k}: {v}")
    print(f"  all chemicals ({len(entity['chems_all_sorted'])}): {entity['chems_all_sorted']}")
    print(f"  strains only in test: {entity['strains_test_only_sorted'][:20]}")
    print(f"  chemicals only in test: {entity['chems_test_only_sorted']}")

    # Per-split entity novelty relative to the train split -- validates S1/S2/S3 semantics
    print("\n  Per-split novelty relative to the 'train' split:")
    split_novelty = {}
    for label, df in (("train_val", meta_tv), ("test", meta_te)):
        for sp, g in df.groupby("split_final"):
            s_new = len(set(g["Strains"].dropna()) - strains_train)
            c_new = len(set(g["perturbation_no_concentration"].dropna()) - chems_train)
            key = f"{label}:{sp}"
            split_novelty[key] = {
                "n_samples": int(len(g)),
                "n_strains": int(g["Strains"].nunique()),
                "n_novel_strains": s_new,
                "n_chems": int(g["perturbation_no_concentration"].nunique()),
                "n_novel_chems": c_new,
            }
            print(f"   {key:<28} n={len(g):<6} strains={g['Strains'].nunique():<4} "
                  f"novel_strains={s_new:<4} chems={g['perturbation_no_concentration'].nunique():<4} "
                  f"novel_chems={c_new}")

    out = {
        "id_checks": id_checks,
        "profile_train_val": prof_tv,
        "profile_test": prof_te,
        "crosstabs": crosstabs,
        "entity_visibility": entity,
        "split_novelty_vs_train": split_novelty,
    }
    dest = RESULTS / "metadata_profile.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved metadata profile -> {dest}")


if __name__ == "__main__":
    main()
