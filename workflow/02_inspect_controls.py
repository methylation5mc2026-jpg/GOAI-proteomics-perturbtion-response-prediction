"""
Step 1b: Characterise the vehicle-control structure to design the delta-matching rule.

The competition scores a fold-change module on Delta = y_treat - y_control, where
controls are the DMSO / Water vehicle samples, and requires the control-matching
rule to be frozen on training data only. This script establishes:

  * which chemicals act as controls vs treatments ('Water', 'DMSO', 'Quality Control');
  * the co-occurrence of pert_id and chemical name (is pert_id a stable chemical key?);
  * how many context keys of increasing coarseness have >= 1 available control, so a
    deterministic fallback hierarchy can be chosen;
  * whether DMSO and Water controls co-exist inside the same context.

Metadata only - no proteome intensities are read.
"""

from __future__ import annotations

from common import load_metadata
from repo_paths import REPO_ROOT, RESULTS_DIR

import json

import numpy as np
import pandas as pd

SESSION = REPO_ROOT
RESULTS = RESULTS_DIR

CONTROL_CHEMS = ("Water", "DMSO")
QC_CHEMS = ("Quality Control",)

np.random.seed(42)

# Candidate context keys, from finest to coarsest. Chosen from the biological +
# measurement covariates present in the metadata.
CONTEXT_LEVELS = [
    ("L1_plate", ["data_source", "Yeast_cell_plate", "Strains", "Medium",
                  "Temperature", "pert_time"]),
    ("L2_plate_notime", ["data_source", "Yeast_cell_plate", "Strains", "Medium",
                         "Temperature"]),
    ("L3_batch_ctx", ["data_source", "Strains", "Medium", "Temperature", "pert_time"]),
    ("L4_batch_ctx_notime", ["data_source", "Strains", "Medium", "Temperature"]),
    ("L5_strain_ctx", ["Strains", "Medium", "Temperature"]),
    ("L6_strain", ["Strains"]),
]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_metadata()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tv, te = load()
    both = pd.concat([tv.assign(_src="train_val"), te.assign(_src="test")],
                     ignore_index=True)

    print("=" * 78)
    print("A. Control / QC sample counts")
    print("=" * 78)
    counts = {}
    for label, df in (("train_val", tv), ("test", te)):
        chem = df["perturbation_no_concentration"]
        n_ctrl = {c: int((chem == c).sum()) for c in CONTROL_CHEMS}
        n_qc = {c: int((chem == c).sum()) for c in QC_CHEMS}
        n_treat = int((~chem.isin(CONTROL_CHEMS + QC_CHEMS)).sum())
        counts[label] = {"controls": n_ctrl, "qc": n_qc, "n_treatment": n_treat,
                         "n_total": int(len(df))}
        print(f"  {label}: controls={n_ctrl} qc={n_qc} treatments={n_treat} total={len(df)}")

    # Which split do control samples live in?
    print("\n  Control samples by split_final:")
    ctrl_mask = both["perturbation_no_concentration"].isin(CONTROL_CHEMS)
    print(pd.crosstab(both.loc[ctrl_mask, "split_final"],
                      both.loc[ctrl_mask, "perturbation_no_concentration"]).to_string())

    print("\n" + "=" * 78)
    print("B. Is pert_id a stable key for the chemical name?")
    print("=" * 78)
    pid_map = both.groupby("pert_id")["perturbation_no_concentration"].nunique()
    chem_map = both.groupby("perturbation_no_concentration")["pert_id"].nunique()
    print(f"  n_pert_id={both['pert_id'].nunique()}  n_chem={both['perturbation_no_concentration'].nunique()}")
    print(f"  pert_id mapping to >1 chemical name: {int((pid_map > 1).sum())}")
    print(f"  chemicals mapping to >1 pert_id   : {int((chem_map > 1).sum())}")
    multi = chem_map[chem_map > 1]
    if len(multi):
        print(f"  chemicals with multiple pert_ids (concentration series):")
        for c, n in multi.items():
            ids = sorted(both.loc[both["perturbation_no_concentration"] == c, "pert_id"].unique())
            print(f"    {c!r}: {n} ids -> {ids}")

    print("\n" + "=" * 78)
    print("C. Control availability per context level (fallback hierarchy design)")
    print("=" * 78)
    # Controls eligible as matching anchors must come from the TRAIN split only
    # (competition rule: control-matching statistics frozen on training data).
    train_ctrl = tv[(tv["split_final"] == "train")
                    & tv["perturbation_no_concentration"].isin(CONTROL_CHEMS)]
    print(f"  Eligible anchor controls (train split, DMSO/Water): {len(train_ctrl)}")

    avail = {}
    for label, df in (("train_val", tv), ("test", te)):
        treat = df[~df["perturbation_no_concentration"].isin(CONTROL_CHEMS + QC_CHEMS)]
        print(f"\n  --- {label}: {len(treat)} treatment samples ---")
        avail[label] = {}
        for lname, keys in CONTEXT_LEVELS:
            ctrl_keys = set(map(tuple, train_ctrl[keys].astype(str).to_numpy()))
            treat_keys = list(map(tuple, treat[keys].astype(str).to_numpy()))
            n_hit = sum(1 for k in treat_keys if k in ctrl_keys)
            pct = 100.0 * n_hit / max(len(treat_keys), 1)
            avail[label][lname] = {"keys": keys, "n_matched": int(n_hit),
                                   "pct_matched": round(pct, 2),
                                   "n_control_groups": len(ctrl_keys)}
            print(f"   {lname:<22} matched {n_hit:>5}/{len(treat_keys)} ({pct:6.2f}%)  "
                  f"control groups={len(ctrl_keys)}")

    print("\n" + "=" * 78)
    print("D. Do DMSO and Water co-exist within the same context group?")
    print("=" * 78)
    keys_l3 = CONTEXT_LEVELS[2][1]
    g = (train_ctrl.groupby(keys_l3)["perturbation_no_concentration"]
         .agg(lambda s: tuple(sorted(set(s)))))
    print("  Vehicle composition of L3_batch_ctx control groups:")
    print(g.value_counts().to_string())

    # Replicate depth of controls per group
    rep = train_ctrl.groupby(keys_l3).size()
    print(f"\n  Control replicates per L3 group: median={rep.median():.1f} "
          f"min={rep.min()} max={rep.max()} n_groups={len(rep)}")

    print("\n" + "=" * 78)
    print("E. Treatment replicate structure (same chemical+context)")
    print("=" * 78)
    treat_tv = tv[~tv["perturbation_no_concentration"].isin(CONTROL_CHEMS + QC_CHEMS)]
    rk = ["Strains", "Medium", "Temperature", "pert_time", "pert_id"]
    rs = treat_tv.groupby(rk).size()
    print(f"  train_val treatment groups (strain x medium x temp x time x pert_id): {len(rs)}")
    print(f"  replicates per group: median={rs.median():.1f} min={rs.min()} max={rs.max()}")
    print(f"  distribution:\n{rs.value_counts().sort_index().to_string()}")

    out = {"control_counts": counts,
           "pert_id_is_unique_per_chem": int((pid_map > 1).sum()) == 0,
           "chems_with_multiple_pert_ids": {c: int(n) for c, n in multi.items()},
           "control_availability_by_level": avail,
           "l3_vehicle_composition": {str(k): int(v) for k, v in g.value_counts().items()},
           "l3_control_replicates": {"median": float(rep.median()),
                                     "min": int(rep.min()), "max": int(rep.max()),
                                     "n_groups": int(len(rep))}}
    dest = RESULTS / "control_structure_profile.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {dest}")


if __name__ == "__main__":
    main()
