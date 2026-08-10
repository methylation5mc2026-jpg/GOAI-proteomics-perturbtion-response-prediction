"""
Step 1c: Refine the control-matching rule and clarify pert_id semantics.

Two problems surfaced in 02_inspect_controls.py:
  (i)  pert_id is not a globally stable chemical key (15 ids map to >1 chemical).
  (ii) Restricting anchor controls to the 'train' split leaves ~58% of test
       treatments unmatched, because the held-out strain (CRD) has no train-split
       control at all.

This script resolves both:
  * tests whether pert_id becomes unique once scoped within data_source;
  * re-computes control availability using the controls that are actually
     present in each released file (train_val controls for train_val samples,
     test controls for test samples) - the competition freezes the matching
     *rule* on training data, not the anchor pool;
  * reports exactly which contexts remain unmatchable at every fallback level.
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

CONTEXT_LEVELS = [
    ("L1_full_ctx", ["data_source", "Strains", "Medium", "Temperature", "pert_time"]),
    ("L2_drop_time", ["data_source", "Strains", "Medium", "Temperature"]),
    ("L3_drop_batch", ["Strains", "Medium", "Temperature", "pert_time"]),
    ("L4_strain_media", ["Strains", "Medium", "Temperature"]),
    ("L5_strain", ["Strains"]),
    ("L6_global", []),
]

np.random.seed(42)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tv, te = load_metadata()
    both = pd.concat([tv, te], ignore_index=True)

    print("=" * 78)
    print("A. pert_id semantics: is it unique per chemical WITHIN data_source?")
    print("=" * 78)
    for src, g in both.groupby("data_source"):
        n_bad = int((g.groupby("pert_id")["perturbation_no_concentration"].nunique() > 1).sum())
        n_chem_multi = int((g.groupby("perturbation_no_concentration")["pert_id"].nunique() > 1).sum())
        print(f"  {src:<12} n_pert_id={g['pert_id'].nunique():<4} "
              f"n_chem={g['perturbation_no_concentration'].nunique():<4} "
              f"ids->multi-chem={n_bad:<3} chems->multi-id={n_chem_multi}")

    # Is (data_source, pert_id) -> chemical a function?
    key_ok = int((both.groupby(["data_source", "pert_id"])
                  ["perturbation_no_concentration"].nunique() > 1).sum())
    print(f"\n  (data_source, pert_id) mapping to >1 chemical: {key_ok}")
    print("  => pert_id is a BATCH-LOCAL plate/dose slot code, not a chemical identity.")
    print("     Chemical entity key for OOD definition must be "
          "'perturbation_no_concentration'.")

    # Does pert_id carry dose information within a chemical?
    print("\n  Chemicals appearing under >1 pert_id within one data_source "
          "(=> dose series):")
    dose_series = {}
    for src, g in both.groupby("data_source"):
        m = g.groupby("perturbation_no_concentration")["pert_id"].nunique()
        for c, n in m[m > 1].items():
            ids = sorted(g.loc[g["perturbation_no_concentration"] == c, "pert_id"].unique())
            dose_series.setdefault(src, {})[c] = ids
            print(f"    {src:<12} {c!r}: {ids}")
    if not dose_series:
        print("    none -> within a batch, chemical and pert_id are 1:1 "
              "(no explicit dose series)")

    print("\n" + "=" * 78)
    print("B. Control availability using in-file controls + fallback hierarchy")
    print("=" * 78)
    avail = {}
    for label, df in (("train_val", tv), ("test", te)):
        ctrl = df[df["perturbation_no_concentration"].isin(CONTROL_CHEMS)]
        treat = df[~df["perturbation_no_concentration"].isin(CONTROL_CHEMS + QC_CHEMS)]
        print(f"\n  --- {label}: {len(treat)} treatments, {len(ctrl)} in-file controls ---")
        assigned = pd.Series(index=treat.index, dtype=object)
        avail[label] = {}
        for lname, keys in CONTEXT_LEVELS:
            todo = assigned[assigned.isna()].index
            if len(todo) == 0:
                avail[label][lname] = {"keys": keys, "n_newly_matched": 0,
                                       "n_control_groups": None}
                print(f"   {lname:<18} (all already matched)")
                continue
            if keys:
                ctrl_keys = set(map(tuple, ctrl[keys].astype(str).to_numpy()))
                tk = list(map(tuple, treat.loc[todo, keys].astype(str).to_numpy()))
                hit = [k in ctrl_keys for k in tk]
                n_groups = len(ctrl_keys)
            else:
                hit = [len(ctrl) > 0] * len(todo)
                n_groups = 1
            newly = [i for i, h in zip(todo, hit) if h]
            assigned.loc[newly] = lname
            cum = int(assigned.notna().sum())
            avail[label][lname] = {"keys": keys, "n_newly_matched": len(newly),
                                   "n_control_groups": n_groups}
            print(f"   {lname:<18} newly matched {len(newly):>5}  "
                  f"cumulative {cum:>5}/{len(treat)} ({100*cum/len(treat):6.2f}%)  "
                  f"control groups={n_groups}")

        unmatched = treat.loc[assigned.isna()]
        print(f"   FINAL unmatched: {len(unmatched)}")
        avail[label]["_level_distribution"] = {
            k: int(v) for k, v in assigned.value_counts(dropna=False).items()
        }

        # Diagnose which contexts fell through the finest level
        fell = treat.loc[assigned != "L1_full_ctx"]
        if len(fell):
            print(f"   Treatments NOT matched at finest L1_full_ctx: {len(fell)}")
            summ = (fell.groupby(["data_source", "Strains", "Temperature", "pert_time"])
                    .size().sort_values(ascending=False))
            print("   top offending (data_source, Strains, Temp, pert_time) contexts:")
            print("   " + summ.head(12).to_string().replace("\n", "\n   "))

    print("\n" + "=" * 78)
    print("C. Where do controls exist? strain x pert_time coverage")
    print("=" * 78)
    for label, df in (("train_val", tv), ("test", te)):
        ctrl = df[df["perturbation_no_concentration"].isin(CONTROL_CHEMS)]
        print(f"\n  --- {label} control counts: Strains x pert_time ---")
        print(pd.crosstab(ctrl["Strains"], ctrl["pert_time"]).to_string())
        print(f"  --- {label} TREATMENT counts: Strains x pert_time ---")
        treat = df[~df["perturbation_no_concentration"].isin(CONTROL_CHEMS + QC_CHEMS)]
        print(pd.crosstab(treat["Strains"], treat["pert_time"]).to_string())

    dest = RESULTS / "matching_rule_diagnostics.json"
    dest.write_text(json.dumps(
        {"pert_id_unique_within_data_source": key_ok == 0,
         "dose_series_within_batch": dose_series,
         "availability_with_fallback": avail}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nSaved -> {dest}")


if __name__ == "__main__":
    main()
