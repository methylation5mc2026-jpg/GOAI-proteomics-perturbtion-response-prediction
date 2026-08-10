"""
Step 3: Control matching and delta (log2 fold-change) computation.

Delta = y_treat - y_control, on median-normalised log2 intensities. The
competition scores a fold-change module on exactly this quantity, so the
matching rule is defined explicitly and frozen here.

Matching rule (frozen)
----------------------
1. Anchor pool = all vehicle-control samples ('Water', 'DMSO'). 'Quality Control'
   samples are EXCLUDED - they are instrument QC injections, not vehicle controls.
2. A treated sample is matched to controls sharing its context, using the first
   level of this hierarchy that yields >= 1 control (see common.CONTEXT_LEVELS):
   L1 (data_source, Strains, Medium, Temperature, pert_time) -> L2 drop time ->
   L3 drop batch -> L4 strain+media -> L5 strain -> L6 global.
   'Yeast_cell_plate' is deliberately NOT a matching key: it was verified to be
   a deterministic function of the L1 key, so it adds no resolution.
3. Control replicates within a matched group are aggregated by MEDIAN (robust to
   the 1-3 replicates available per group).
4. Primary vehicle policy = POOLED (DMSO and Water controls pooled), because the
   released metadata contains no per-chemical vehicle annotation and 324/381
   context groups contain both vehicles. DMSO-only and Water-only variants are
   computed as a sensitivity analysis.
5. Anchor pool per file: controls released in the same file. train_val controls
   are additionally offered to test samples of *seen* strains, since the held-out
   strain CRD is the only strain with dedicated test controls and seen-strain
   test controls are near-absent by design.

Outputs
-------
workflow/processed_delta_matrix.parquet      (train_val treated samples, pooled vehicle)
workflow/processed_delta_matrix_test.parquet (test treated samples, pooled vehicle)
results/delta_matching_report.csv, results/delta_summary.json
figures/eda_delta_distribution.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CHEM_COL, CONTEXT_LEVELS, CONTROL_CHEMS, DATA, FIGURES,
                    ID_COL, QC_CHEMS, RESULTS, SEED, WORKFLOW)

np.random.seed(SEED)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.linewidth": 0.6, "figure.dpi": 110})

# Proteins never detected in a control group give an all-NaN slice; the resulting
# NaN is the intended encoding ("delta undefined"), so the warning is noise.
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")


def load_all() -> tuple[pd.DataFrame, np.ndarray, list[str], pd.DataFrame, np.ndarray]:
    """Load normalised log2 matrices and annotated metadata for both files."""
    tv = pd.read_parquet(WORKFLOW / "processed_train_val_proteome.parquet")
    te = pd.read_parquet(WORKFLOW / "processed_test_proteome.parquet")
    proteins = [c for c in tv.columns if c != ID_COL]
    meta_tv = pd.read_csv(DATA / "meta_train_val_annotated.csv", dtype=str)
    meta_te = pd.read_csv(DATA / "meta_test_annotated.csv", dtype=str)
    m_tv = tv[proteins].to_numpy(dtype="float32")
    m_te = te[proteins].to_numpy(dtype="float32")
    assert (tv[ID_COL].astype(str).to_numpy() == meta_tv[ID_COL].to_numpy()).all()
    assert (te[ID_COL].astype(str).to_numpy() == meta_te[ID_COL].to_numpy()).all()
    return meta_tv, m_tv, proteins, meta_te, m_te


def build_control_index(meta_pool: pd.DataFrame, mat_pool: np.ndarray,
                        vehicle: str) -> dict[str, dict[tuple, np.ndarray]]:
    """Pre-aggregate control profiles per context level.

    Parameters
    ----------
    meta_pool : pandas.DataFrame
        Metadata of the anchor control pool.
    mat_pool : numpy.ndarray
        Matching ``(n_controls, n_proteins)`` normalised log2 matrix.
    vehicle : {'pooled', 'DMSO', 'Water'}
        Vehicle policy.

    Returns
    -------
    dict
        level name -> {context key tuple -> median control profile}
    """
    if vehicle != "pooled":
        keep = (meta_pool[CHEM_COL] == vehicle).to_numpy()
        meta_pool, mat_pool = meta_pool.loc[keep].reset_index(drop=True), mat_pool[keep]

    index: dict[str, dict[tuple, np.ndarray]] = {}
    for lname, keys in CONTEXT_LEVELS:
        d: dict[tuple, np.ndarray] = {}
        if keys:
            gk = list(map(tuple, meta_pool[keys].astype(str).to_numpy()))
        else:
            gk = [("__global__",)] * len(meta_pool)
        groups: dict[tuple, list[int]] = {}
        for i, k in enumerate(gk):
            groups.setdefault(k, []).append(i)
        for k, idx in groups.items():
            with np.errstate(all="ignore"):
                d[k] = np.nanmedian(mat_pool[idx], axis=0).astype("float32")
        index[lname] = d
    return index


def compute_delta(meta_treat: pd.DataFrame, mat_treat: np.ndarray,
                  ctrl_index: dict, label: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute Delta = y_treat - y_control_matched with the frozen hierarchy.

    Returns
    -------
    delta : numpy.ndarray
        ``(n_treat, n_proteins)`` float32; NaN where treat or control is missing.
    report : pandas.DataFrame
        Per-sample matching level, control replicate count, and delta coverage.
    """
    n, p = mat_treat.shape
    delta = np.full((n, p), np.nan, dtype="float32")
    levels_used = np.empty(n, dtype=object)
    n_ctrl_used = np.zeros(n, dtype=int)

    # Pre-compute the context key of every treated sample at every level
    keyed = {}
    for lname, keys in CONTEXT_LEVELS:
        if keys:
            keyed[lname] = list(map(tuple, meta_treat[keys].astype(str).to_numpy()))
        else:
            keyed[lname] = [("__global__",)] * n

    for i in range(n):
        for lname, _ in CONTEXT_LEVELS:
            prof = ctrl_index[lname].get(keyed[lname][i])
            if prof is not None:
                delta[i] = mat_treat[i] - prof
                levels_used[i] = lname
                n_ctrl_used[i] = 1
                break
        if (i + 1) % 1000 == 0:
            print(f"    [{label}] matched {i+1}/{n} treated samples ...", flush=True)

    report = pd.DataFrame({
        ID_COL: meta_treat[ID_COL].to_numpy(),
        "file": label,
        "match_level": levels_used,
        "n_delta_observed": np.isfinite(delta).sum(axis=1),
        "frac_delta_observed": np.isfinite(delta).mean(axis=1),
    })
    for c in ["data_source", "Strains", "Medium", "Temperature", "pert_time",
              "split_final", CHEM_COL]:
        report[c] = meta_treat[c].to_numpy()
    return delta, report


def main() -> None:
    for d in (RESULTS, FIGURES, WORKFLOW):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading normalised matrices ...")
    meta_tv, m_tv, proteins, meta_te, m_te = load_all()

    is_ctrl_tv = meta_tv[CHEM_COL].isin(CONTROL_CHEMS).to_numpy()
    is_qc_tv = meta_tv[CHEM_COL].isin(QC_CHEMS).to_numpy()
    is_ctrl_te = meta_te[CHEM_COL].isin(CONTROL_CHEMS).to_numpy()
    is_qc_te = meta_te[CHEM_COL].isin(QC_CHEMS).to_numpy()

    # Anchor pools -------------------------------------------------------
    # train_val: its own controls (fully covers all 5 seen strains x 6 times).
    pool_tv_meta = meta_tv.loc[is_ctrl_tv].reset_index(drop=True)
    pool_tv_mat = m_tv[is_ctrl_tv]
    # test: its own controls (covers novel strain CRD) + train_val controls
    # (needed for seen strains, which have almost no dedicated test controls).
    pool_te_meta = pd.concat([meta_te.loc[is_ctrl_te], meta_tv.loc[is_ctrl_tv]],
                             ignore_index=True)
    pool_te_mat = np.vstack([m_te[is_ctrl_te], m_tv[is_ctrl_tv]])
    print(f"  anchor pool train_val: {len(pool_tv_meta)} controls")
    print(f"  anchor pool test     : {len(pool_te_meta)} controls "
          f"({int(is_ctrl_te.sum())} in-file + {int(is_ctrl_tv.sum())} from train_val)")

    treat_tv_meta = meta_tv.loc[~(is_ctrl_tv | is_qc_tv)].reset_index(drop=True)
    treat_tv_mat = m_tv[~(is_ctrl_tv | is_qc_tv)]
    treat_te_meta = meta_te.loc[~(is_ctrl_te | is_qc_te)].reset_index(drop=True)
    treat_te_mat = m_te[~(is_ctrl_te | is_qc_te)]
    print(f"  treated samples: train_val={len(treat_tv_meta)}, test={len(treat_te_meta)}")
    print(f"  'Quality Control' injections excluded: "
          f"train_val={int(is_qc_tv.sum())}, test={int(is_qc_te.sum())}")

    print("\n[2/6] Computing PRIMARY delta (pooled vehicle, median aggregation) ...")
    idx_tv = build_control_index(pool_tv_meta, pool_tv_mat, "pooled")
    idx_te = build_control_index(pool_te_meta, pool_te_mat, "pooled")
    d_tv, rep_tv = compute_delta(treat_tv_meta, treat_tv_mat, idx_tv, "train_val")
    d_te, rep_te = compute_delta(treat_te_meta, treat_te_mat, idx_te, "test")

    report = pd.concat([rep_tv, rep_te], ignore_index=True)
    report.to_csv(RESULTS / "delta_matching_report.csv", index=False)
    print("\n  match-level distribution:")
    print(pd.crosstab(report["file"], report["match_level"]).to_string())
    print("\n  delta coverage (fraction of proteins with a defined delta):")
    print(report.groupby("file")["frac_delta_observed"]
          .describe()[["mean", "50%", "min", "max"]].round(4).to_string())
    n_unmatched = int(report["match_level"].isna().sum())
    print(f"  unmatched treated samples: {n_unmatched}")

    print("\n[3/6] Vehicle sensitivity analysis (pooled vs DMSO-only vs Water-only) ...")
    sens = {}
    for veh in ("DMSO", "Water"):
        idx_v = build_control_index(pool_tv_meta, pool_tv_mat, veh)
        d_v, rep_v = compute_delta(treat_tv_meta, treat_tv_mat, idx_v, f"tv_{veh}")
        ok = np.isfinite(d_tv) & np.isfinite(d_v)
        # per-sample PCC between pooled delta and vehicle-specific delta
        pccs = []
        for i in range(d_tv.shape[0]):
            m = ok[i]
            if m.sum() > 50:
                a, b = d_tv[i, m], d_v[i, m]
                if a.std() > 0 and b.std() > 0:
                    pccs.append(float(np.corrcoef(a, b)[0, 1]))
        pccs = np.array(pccs)
        sens[veh] = {"n_samples_compared": int(len(pccs)),
                     "pcc_vs_pooled_median": round(float(np.median(pccs)), 4),
                     "pcc_vs_pooled_q05": round(float(np.percentile(pccs, 5)), 4),
                     "pcc_vs_pooled_q95": round(float(np.percentile(pccs, 95)), 4),
                     "level_distribution": {str(k): int(v) for k, v in
                                            rep_v["match_level"].value_counts().items()}}
        print(f"  {veh}-only vs pooled: per-sample PCC median="
              f"{sens[veh]['pcc_vs_pooled_median']:.4f} "
              f"[q05={sens[veh]['pcc_vs_pooled_q05']:.4f}, "
              f"q95={sens[veh]['pcc_vs_pooled_q95']:.4f}]")
        del d_v

    print("\n[4/6] Control-replicate aggregation sensitivity (median vs mean) ...")
    idx_mean = {}
    for lname, keys in CONTEXT_LEVELS:
        d = {}
        gk = (list(map(tuple, pool_tv_meta[keys].astype(str).to_numpy())) if keys
              else [("__global__",)] * len(pool_tv_meta))
        groups: dict[tuple, list[int]] = {}
        for i, k in enumerate(gk):
            groups.setdefault(k, []).append(i)
        with np.errstate(all="ignore"):
            for k, ii in groups.items():
                d[k] = np.nanmean(pool_tv_mat[ii], axis=0).astype("float32")
        idx_mean[lname] = d
    d_mean, _ = compute_delta(treat_tv_meta, treat_tv_mat, idx_mean, "tv_mean")
    ok = np.isfinite(d_tv) & np.isfinite(d_mean)
    diffs = np.abs(d_tv[ok] - d_mean[ok])
    agg_sens = {"median_abs_diff": round(float(np.median(diffs)), 5),
                "q95_abs_diff": round(float(np.percentile(diffs, 95)), 5),
                "corr_all_cells": round(float(np.corrcoef(d_tv[ok], d_mean[ok])[0, 1]), 5)}
    print(f"  |median-agg delta - mean-agg delta|: median={agg_sens['median_abs_diff']}, "
          f"q95={agg_sens['q95_abs_diff']}, overall corr={agg_sens['corr_all_cells']}")
    del d_mean

    print("\n[5/6] Delta distribution statistics ...")
    stats = {}
    for label, dm, rp in (("train_val", d_tv, rep_tv), ("test", d_te, rep_te)):
        vals = dm[np.isfinite(dm)]
        stats[label] = {
            "n_samples": int(dm.shape[0]),
            "n_defined_cells": int(vals.size),
            "pct_defined": round(100 * vals.size / dm.size, 3),
            "mean": round(float(vals.mean()), 5),
            "median": round(float(np.median(vals)), 5),
            "sd": round(float(vals.std()), 5),
            "mad": round(float(np.median(np.abs(vals - np.median(vals)))), 5),
            "pct_abs_gt_1": round(100 * float((np.abs(vals) > 1).mean()), 3),
            "percentiles": {str(q): round(float(np.percentile(vals, q)), 4)
                            for q in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)},
        }
        print(f"  {label}: median={stats[label]['median']:.4f} sd={stats[label]['sd']:.4f} "
              f"MAD={stats[label]['mad']:.4f} |delta|>1 in {stats[label]['pct_abs_gt_1']}% of cells")

    # Per-chemical effect magnitude (train_val), a biology sanity check
    per_chem = []
    with np.errstate(all="ignore"):
        for chem, g in treat_tv_meta.groupby(CHEM_COL):
            sub = d_tv[g.index.to_numpy()]
            v = sub[np.isfinite(sub)]
            if v.size < 100:
                continue
            per_chem.append({"chemical": chem, "n_samples": int(len(g)),
                             "median_abs_delta": float(np.median(np.abs(v))),
                             "frac_abs_gt_1": float((np.abs(v) > 1).mean()),
                             "sd_delta": float(v.std())})
    per_chem = pd.DataFrame(per_chem).sort_values("median_abs_delta", ascending=False)
    per_chem.to_csv(RESULTS / "delta_per_chemical.csv", index=False)
    print("\n  strongest perturbations by median |delta| (train_val):")
    print(per_chem.head(10).to_string(index=False))
    print("\n  weakest perturbations:")
    print(per_chem.tail(5).to_string(index=False))

    print("\n[6/6] Figure + export ...")
    _figure(d_tv, treat_tv_meta, per_chem, report, sens)

    ids_tv = treat_tv_meta[ID_COL]
    ids_te = treat_te_meta[ID_COL]
    _save(ids_tv, d_tv, proteins, WORKFLOW / "processed_delta_matrix.parquet",
          treat_tv_meta, rep_tv)
    _save(ids_te, d_te, proteins, WORKFLOW / "processed_delta_matrix_test.parquet",
          treat_te_meta, rep_te)

    out = {"matching_rule": {
                "anchor_chemicals": list(CONTROL_CHEMS),
                "excluded_from_anchors": list(QC_CHEMS),
                "hierarchy": [{"level": n, "keys": k} for n, k in CONTEXT_LEVELS],
                "replicate_aggregation": "median",
                "primary_vehicle_policy": "pooled (DMSO + Water)",
                "plate_excluded_reason": "Yeast_cell_plate is a deterministic "
                                         "function of the L1 key; adds no resolution",
                "test_anchor_pool": "in-file test controls + train_val controls"},
           "match_level_counts": {f"{f}:{lv}": int(v) for (f, lv), v in
                                  report.groupby(["file", "match_level"]).size().items()},
           "n_unmatched": n_unmatched,
           "delta_statistics": stats,
           "vehicle_sensitivity": sens,
           "aggregation_sensitivity_median_vs_mean": agg_sens}
    (RESULTS / "delta_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("Saved results/delta_summary.json")


def _save(ids, mat, proteins, dest, meta, rep) -> None:
    df = pd.DataFrame(mat, columns=proteins)
    df.insert(0, ID_COL, np.asarray(ids, dtype=object))
    df.insert(1, "match_level", rep["match_level"].to_numpy())
    df.insert(2, "split_final", meta["split_final"].to_numpy())
    df.insert(3, CHEM_COL, meta[CHEM_COL].to_numpy())
    df.to_parquet(dest, index=False, compression="snappy")
    print(f"  [save] {dest.name}: {df.shape[0]} x {df.shape[1]} "
          f"-> {dest.stat().st_size / 1e6:.1f} MB")


def _figure(d_tv, meta_treat, per_chem, report, sens) -> None:
    """Four-panel delta diagnostics."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # A: global delta distribution
    ax = axes[0, 0]
    vals = d_tv[np.isfinite(d_tv)]
    samp = vals[rng.choice(vals.size, size=min(500_000, vals.size), replace=False)]
    ax.hist(samp, bins=200, color="#3b6ea5")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    for t in (-1, 1):
        ax.axvline(t, color="#c1666b", lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xlim(-6, 6)
    ax.set_xlabel(r"$\Delta$ = log2 FC (treated - matched control)")
    ax.set_ylabel("count (log scale)")
    ax.set_title(r"A  Global $\Delta$ distribution (train_val)" "\n"
                 "dotted lines: |delta| = 1 DEP threshold", loc="left")

    # B: delta spread by strain
    ax = axes[0, 1]
    strains = sorted(meta_treat["Strains"].unique())
    data = []
    for s in strains:
        idx = meta_treat.index[meta_treat["Strains"] == s].to_numpy()
        sub = d_tv[idx]
        v = sub[np.isfinite(sub)]
        data.append(v[rng.choice(v.size, size=min(120_000, v.size), replace=False)])
    bp = ax.boxplot(data, tick_labels=strains, showfliers=False, patch_artist=True,
                    widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8ab4d8")
        patch.set_linewidth(0.6)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_ylabel(r"$\Delta$ (log2 FC)")
    ax.set_title("B  Perturbation response by strain", loc="left")

    # C: per-chemical effect magnitude
    ax = axes[1, 0]
    top = per_chem.head(22).iloc[::-1]
    ax.barh(np.arange(len(top)), top["median_abs_delta"], color="#c1666b", height=0.7)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([c[:26] for c in top["chemical"]], fontsize=5.5)
    ax.set_xlabel(r"median $|\Delta|$ across proteins")
    ax.set_title("C  Strongest perturbations (train_val)", loc="left")

    # D: per-sample delta coverage. Every treated sample matched at the finest
    # level L1, so a match-level bar chart would be a single degenerate bar;
    # coverage is the informative quantity (a delta needs the protein observed
    # in BOTH the treated sample and its matched control).
    ax = axes[1, 1]
    for label, color in (("train_val", "#3b6ea5"), ("test", "#c1666b")):
        sub = report[report["file"] == label]
        ax.hist(sub["frac_delta_observed"], bins=60, alpha=0.65, label=label,
                color=color)
    ax.set_xlabel(r"fraction of proteins with a defined $\Delta$")
    ax.set_ylabel("treated samples")
    ax.set_title(r"D  Per-sample $\Delta$ coverage", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ct = report.groupby("file")["match_level"].agg(
        lambda s: s.value_counts().idxmax() if len(s) else "none")
    txt = ("all treated samples matched at finest level\n"
           f"L1 (batch,strain,medium,temp,time): "
           f"{int((report['match_level'] == 'L1_full_ctx').sum())}/{len(report)}\n"
           f"vehicle sensitivity (train_val, per-sample PCC):\n"
           f"  DMSO-only vs pooled = {sens['DMSO']['pcc_vs_pooled_median']:.3f}\n"
           f"  Water-only vs pooled = {sens['Water']['pcc_vs_pooled_median']:.3f}")
    ax.text(0.02, 0.62, txt, transform=ax.transAxes, ha="left", va="top", fontsize=5.8)

    fig.tight_layout()
    fig.savefig(FIGURES / "eda_delta_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "eda_delta_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  saved figures/eda_delta_distribution.png")


if __name__ == "__main__":
    main()
