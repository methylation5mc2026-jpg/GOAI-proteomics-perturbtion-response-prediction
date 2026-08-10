"""
Step 5b: Independent verification of the exported artefacts.

Re-derives key claims straight from the parquet files rather than trusting the
in-memory state of earlier steps, to catch export bugs (wrong ordering, silent
dtype coercion, NaN loss, delta not equal to treat - control).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (CHEM_COL, CONTROL_CHEMS, DATA, ID_COL, RESULTS, SEED,
                    WORKFLOW)

np.random.seed(SEED)
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def main() -> None:
    print("[1/4] Loading exported artefacts ...")
    prot = pd.read_parquet(WORKFLOW / "processed_train_val_proteome.parquet")
    delta = pd.read_parquet(WORKFLOW / "processed_delta_matrix.parquet")
    meta = pd.read_csv(DATA / "meta_train_val_annotated.csv", dtype=str)
    proteins = [c for c in prot.columns if c != ID_COL]
    ann = [ID_COL, "match_level", "split_final", CHEM_COL]
    d_prot = [c for c in delta.columns if c not in ann]

    print("[2/4] Structural checks ...")
    check("proteome protein columns == 5243", len(proteins) == 5243, f"{len(proteins)}")
    check("delta protein columns match proteome order", d_prot == proteins)
    check("proteome sample_ID order == metadata order",
          (prot[ID_COL].astype(str).to_numpy() == meta[ID_COL].to_numpy()).all())
    check("no duplicated sample_ID in proteome", not prot[ID_COL].duplicated().any())
    check("no duplicated sample_ID in delta", not delta[ID_COL].duplicated().any())
    check("delta excludes all control samples",
          not delta[CHEM_COL].isin(CONTROL_CHEMS).any())
    check("delta excludes Quality Control",
          not (delta[CHEM_COL] == "Quality Control").any())
    check("all delta rows matched at L1_full_ctx",
          (delta["match_level"] == "L1_full_ctx").all(),
          f"levels={delta['match_level'].unique().tolist()}")

    print("[3/4] Numeric integrity ...")
    pm = prot[proteins].to_numpy(dtype="float32")
    dm = delta[d_prot].to_numpy(dtype="float32")
    check("proteome retains NaN (missingness not silently filled)",
          bool(np.isnan(pm).any()),
          f"{100*np.isnan(pm).mean():.2f}% NaN")
    check("no infinite values in proteome", not bool(np.isinf(pm).any()))
    check("no infinite values in delta", not bool(np.isinf(dm).any()))
    check("delta centred near zero",
          abs(float(np.nanmedian(dm))) < 0.05, f"median={np.nanmedian(dm):.5f}")

    # Re-derive delta for a random sample of rows, straight from the parquet
    print("  re-deriving Delta from scratch for 40 random treated samples ...")
    is_ctrl = meta[CHEM_COL].isin(CONTROL_CHEMS).to_numpy()
    ctrl_meta = meta.loc[is_ctrl].reset_index(drop=True)
    ctrl_mat = pm[is_ctrl]
    keys = ["data_source", "Strains", "Medium", "Temperature", "pert_time"]
    ck = list(map(tuple, ctrl_meta[keys].astype(str).to_numpy()))
    groups: dict[tuple, list[int]] = {}
    for i, k in enumerate(ck):
        groups.setdefault(k, []).append(i)

    pos = {s: i for i, s in enumerate(prot[ID_COL].astype(str))}
    mrow = {s: i for i, s in enumerate(meta[ID_COL])}
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(delta), size=40, replace=False)
    max_dev, n_cmp = 0.0, 0
    for r in pick:
        sid = str(delta[ID_COL].iloc[r])
        k = tuple(meta.loc[mrow[sid], keys].astype(str))
        prof = np.nanmedian(ctrl_mat[groups[k]], axis=0)
        expect = pm[pos[sid]] - prof
        got = dm[r]
        ok = np.isfinite(expect) & np.isfinite(got)
        if ok.sum():
            max_dev = max(max_dev, float(np.abs(expect[ok] - got[ok]).max()))
            n_cmp += int(ok.sum())
        # NaN pattern must agree exactly
        if not (np.isfinite(expect) == np.isfinite(got)).all():
            check(f"NaN pattern mismatch for {sid}", False)
            break
    check("Delta == y_treat - median(matched controls), re-derived",
          max_dev < 1e-4, f"max abs deviation={max_dev:.2e} over {n_cmp} cells")

    print("[4/4] Consistency with reported summary ...")
    summ = json.loads((RESULTS / "qc_summary.json").read_text(encoding="utf-8"))
    rep_pct = summ["missingness"]["train_val"]["pct_missing_nan"]
    log_tv = pd.read_parquet(DATA / "log2_train_val.parquet")
    obs_pct = 100 * float(np.isnan(
        log_tv[[c for c in log_tv.columns if c != ID_COL]]
        .to_numpy(dtype="float32")).mean())
    check("reported missingness matches log2 export",
          abs(rep_pct - obs_pct) < 0.01, f"reported={rep_pct}, observed={obs_pct:.3f}")
    check("all manifest outputs exist",
          json.loads((WORKFLOW.parent / "manifest.json")
                     .read_text(encoding="utf-8"))["n_missing"] == 0)

    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    print(f"\n{'=' * 60}\nVERIFICATION: {len(CHECKS) - n_fail}/{len(CHECKS)} passed, "
          f"{n_fail} failed\n{'=' * 60}")
    (RESULTS / "verification_report.json").write_text(json.dumps(
        {"n_checks": len(CHECKS), "n_failed": n_fail,
         "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in CHECKS]},
        indent=2), encoding="utf-8")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
