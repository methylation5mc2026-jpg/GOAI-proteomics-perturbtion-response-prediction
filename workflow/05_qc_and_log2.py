"""
Step 2: Missingness profiling, QC metrics, log2 transform and normalised export.

Outputs
-------
data/log2_train_val.parquet, data/log2_test.parquet
    log2 intensities, NaN preserved (no imputation).
workflow/processed_train_val_proteome.parquet, workflow/processed_test_proteome.parquet
    Median-normalised log2 intensities (per-sample median centring, re-centred on
    a global offset frozen from the 'train' split only), NaN preserved.
results/qc_metrics_samples.csv, results/qc_metrics_proteins.csv
figures/eda_missing_values.png, figures/eda_intensity_distribution.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CHEM_COL, DATA, FIGURES, ID_COL, PROT_TE, PROT_TV, RESULTS,
                    SEED, WORKFLOW, align_to_metadata, load_metadata,
                    load_proteome, log2_transform, sample_role,
                    save_matrix_parquet)

np.random.seed(SEED)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.linewidth": 0.6, "figure.dpi": 110})


def main() -> None:
    for d in (DATA, RESULTS, FIGURES, WORKFLOW):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading metadata and proteome matrices ...")
    meta_tv, meta_te = load_metadata()
    ids_tv, raw_tv, proteins = load_proteome(PROT_TV, "train_val")
    ids_te, raw_te, proteins_te = load_proteome(PROT_TE, "test")
    if proteins != proteins_te:
        raise ValueError("protein column order differs between train_val and test")
    meta_tv, raw_tv = align_to_metadata(ids_tv, raw_tv, meta_tv, "train_val")
    meta_te, raw_te = align_to_metadata(ids_te, raw_te, meta_te, "test")
    n_prot = len(proteins)

    meta_tv["sample_role"] = sample_role(meta_tv[CHEM_COL])
    meta_te["sample_role"] = sample_role(meta_te[CHEM_COL])
    print(f"  role counts train_val: {meta_tv['sample_role'].value_counts().to_dict()}")
    print(f"  role counts test     : {meta_te['sample_role'].value_counts().to_dict()}")

    print("\n[2/7] Missingness profiling (NaN vs exact zero kept distinct) ...")
    miss = {}
    for label, mat in (("train_val", raw_tv), ("test", raw_te)):
        obs = np.isfinite(mat)
        n_cells = mat.size
        n_nan = int((~obs).sum())
        n_zero = int((mat == 0).sum())
        miss[label] = {
            "n_samples": int(mat.shape[0]), "n_proteins": int(mat.shape[1]),
            "n_cells": int(n_cells),
            "n_missing_nan": n_nan, "pct_missing_nan": round(100 * n_nan / n_cells, 3),
            "n_exact_zero": n_zero, "pct_exact_zero": round(100 * n_zero / n_cells, 6),
            "detected_per_sample_median": float(np.median(obs.sum(axis=1))),
            "detected_per_sample_min": int(obs.sum(axis=1).min()),
            "detected_per_sample_max": int(obs.sum(axis=1).max()),
            "protein_detection_rate_median": float(np.median(obs.mean(axis=0))),
            "n_proteins_detected_in_all": int((obs.mean(axis=0) == 1.0).sum()),
            "n_proteins_never_detected": int((obs.mean(axis=0) == 0.0).sum()),
        }
        for k, v in miss[label].items():
            print(f"  {label}.{k} = {v}")

    # Per-protein detection rates
    det_tv = np.isfinite(raw_tv).mean(axis=0)
    det_te = np.isfinite(raw_te).mean(axis=0)
    for thr in (0.5, 0.7, 0.9, 0.95, 0.99, 1.0):
        n_both = int(((det_tv >= thr) & (det_te >= thr)).sum())
        print(f"  proteins with detection rate >= {thr:.2f} in BOTH files: {n_both}")
        miss[f"n_proteins_detrate_ge_{thr:g}_both"] = n_both

    print("\n[3/7] log2 transform (NaN preserved) ...")
    log_tv = log2_transform(raw_tv)
    log_te = log2_transform(raw_te)
    del raw_tv, raw_te

    print("\n[4/7] Median normalisation with offsets frozen on the 'train' split ...")
    # Per-sample median of observed log2 values
    with np.errstate(all="ignore"):
        samp_med_tv = np.nanmedian(log_tv, axis=1)
        samp_med_te = np.nanmedian(log_te, axis=1)
    train_mask = (meta_tv["split_final"] == "train").to_numpy()
    ref_offset = float(np.nanmedian(samp_med_tv[train_mask]))
    print(f"  reference offset (median of per-sample medians, TRAIN split only) "
          f"= {ref_offset:.4f}")
    print(f"  per-sample median spread train_val: "
          f"{np.nanpercentile(samp_med_tv, [1, 50, 99]).round(3).tolist()}")
    print(f"  per-sample median spread test     : "
          f"{np.nanpercentile(samp_med_te, [1, 50, 99]).round(3).tolist()}")

    norm_tv = log_tv - samp_med_tv[:, None] + ref_offset
    norm_te = log_te - samp_med_te[:, None] + ref_offset

    print("\n[5/7] Per-sample and per-protein QC tables ...")
    qc_rows = []
    for label, meta, logm, normm, smed in (
            ("train_val", meta_tv, log_tv, norm_tv, samp_med_tv),
            ("test", meta_te, log_te, norm_te, samp_med_te)):
        obs = np.isfinite(logm)
        with np.errstate(all="ignore"):
            q = pd.DataFrame({
                ID_COL: meta[ID_COL].to_numpy(),
                "file": label,
                "data_source": meta["data_source"].to_numpy(),
                "Strains": meta["Strains"].to_numpy(),
                "Medium": meta["Medium"].to_numpy(),
                "Temperature": meta["Temperature"].to_numpy(),
                "pert_time": meta["pert_time"].to_numpy(),
                "instrument": meta["instrument"].to_numpy(),
                "Yeast_cell_plate": meta["Yeast_cell_plate"].to_numpy(),
                "split_final": meta["split_final"].to_numpy(),
                "chemical": meta[CHEM_COL].to_numpy(),
                "sample_role": meta["sample_role"].to_numpy(),
                "n_detected": obs.sum(axis=1),
                "frac_detected": obs.mean(axis=1),
                "log2_median": smed,
                "log2_mean": np.nanmean(logm, axis=1),
                "log2_sd": np.nanstd(logm, axis=1),
                "log2_iqr": (np.nanpercentile(logm, 75, axis=1)
                             - np.nanpercentile(logm, 25, axis=1)),
            })
        qc_rows.append(q)
    qc_samples = pd.concat(qc_rows, ignore_index=True)
    qc_samples.to_csv(RESULTS / "qc_metrics_samples.csv", index=False)
    print(f"  saved qc_metrics_samples.csv ({qc_samples.shape})")

    # Outlier flagging by detection depth (robust MAD rule, diagnostic only)
    d = qc_samples["n_detected"].to_numpy(dtype=float)
    med, mad = np.median(d), np.median(np.abs(d - np.median(d)))
    rob = 0.6745 * (d - med) / mad if mad > 0 else np.zeros_like(d)
    qc_samples["detection_robust_z"] = rob
    n_low = int((rob < -3.5).sum())
    print(f"  detection-depth outliers (robust z < -3.5, FLAGGED not removed): {n_low}")

    with np.errstate(all="ignore"):
        qc_proteins = pd.DataFrame({
            "protein": proteins,
            "det_rate_train_val": det_tv,
            "det_rate_test": det_te,
            "log2_mean_train_val": np.nanmean(log_tv, axis=0),
            "log2_sd_train_val": np.nanstd(log_tv, axis=0),
            "log2_mean_test": np.nanmean(log_te, axis=0),
            "log2_sd_test": np.nanstd(log_te, axis=0),
        })
    qc_proteins.to_csv(RESULTS / "qc_metrics_proteins.csv", index=False)
    print(f"  saved qc_metrics_proteins.csv ({qc_proteins.shape})")

    print("\n[6/7] Figures ...")
    _fig_missing(qc_samples, det_tv, det_te, log_tv, meta_tv, n_prot)
    _fig_intensity(log_tv, norm_tv, log_te, samp_med_tv, meta_tv)

    print("\n[7/7] Exporting matrices ...")
    save_matrix_parquet(meta_tv[ID_COL], log_tv, proteins, DATA / "log2_train_val.parquet")
    save_matrix_parquet(meta_te[ID_COL], log_te, proteins, DATA / "log2_test.parquet")
    save_matrix_parquet(meta_tv[ID_COL], norm_tv, proteins,
                        WORKFLOW / "processed_train_val_proteome.parquet")
    save_matrix_parquet(meta_te[ID_COL], norm_te, proteins,
                        WORKFLOW / "processed_test_proteome.parquet")

    # Persist metadata with derived role column for later steps
    meta_tv.to_csv(DATA / "meta_train_val_annotated.csv", index=False)
    meta_te.to_csv(DATA / "meta_test_annotated.csv", index=False)

    stats = {"missingness": miss,
             "normalisation": {"method": "per-sample median centring of log2, "
                                         "re-centred on train-split reference offset",
                               "reference_offset_log2": ref_offset,
                               "frozen_on": "split_final == 'train' of train_val"},
             "detection_outliers_flagged_robust_z_lt_-3.5": n_low,
             "n_proteins": n_prot}
    (RESULTS / "qc_stats_partial.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nDone. Wrote results/qc_stats_partial.json")


def _fig_missing(qc: pd.DataFrame, det_tv, det_te, log_tv, meta_tv, n_prot) -> None:
    """Missingness diagnostics: 4 panels."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))

    ax = axes[0, 0]
    bins = np.linspace(0, 1, 60)
    ax.hist(det_tv, bins=bins, alpha=0.65, label="train_val", color="#3b6ea5")
    ax.hist(det_te, bins=bins, alpha=0.65, label="test", color="#c1666b")
    ax.set_xlabel("per-protein detection rate")
    ax.set_ylabel("number of proteins")
    ax.set_title(f"A  Protein detection rate ({n_prot} proteins)", loc="left")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for label, color in (("train_val", "#3b6ea5"), ("test", "#c1666b")):
        sub = qc[qc["file"] == label]
        ax.hist(sub["n_detected"], bins=50, alpha=0.65, label=label, color=color)
    ax.set_xlabel("proteins detected per sample")
    ax.set_ylabel("number of samples")
    ax.set_title("B  Detection depth per sample", loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    order = sorted(qc["data_source"].unique())
    data = [qc.loc[qc["data_source"] == b, "frac_detected"].to_numpy() for b in order]
    bp = ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True,
                    widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8ab4d8")
        patch.set_linewidth(0.6)
    ax.set_ylabel("fraction of proteins detected")
    ax.set_title("C  Detection completeness by batch", loc="left")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 1]
    # Missingness vs abundance -> left-censoring signature
    with np.errstate(all="ignore"):
        mean_ab = np.nanmean(log_tv, axis=0)
    ax.scatter(mean_ab, det_tv, s=2, alpha=0.25, color="#444444", rasterized=True)
    ax.set_xlabel("mean observed log2 intensity (train_val)")
    ax.set_ylabel("detection rate")
    ax.set_title("D  Missingness is abundance-dependent\n(left-censoring signature)",
                 loc="left")

    fig.tight_layout()
    fig.savefig(FIGURES / "eda_missing_values.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "eda_missing_values.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  saved figures/eda_missing_values.png")


def _fig_intensity(log_tv, norm_tv, log_te, samp_med_tv, meta_tv) -> None:
    """Intensity distributions before/after log2 and before/after normalisation."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))

    # Panel A: linear vs log2 for a random cell subsample
    idx_r = rng.choice(log_tv.shape[0], size=min(600, log_tv.shape[0]), replace=False)
    sub = log_tv[idx_r]
    obs = sub[np.isfinite(sub)]
    lin = np.power(2.0, obs[rng.choice(obs.size, size=min(300_000, obs.size),
                                       replace=False)])
    ax = axes[0, 0]
    ax.hist(lin, bins=np.logspace(np.log10(max(lin.min(), 1)), np.log10(lin.max()), 60),
            color="#8c6d31")
    ax.set_xscale("log")
    ax.set_xlabel("raw intensity (log-spaced bins)")
    ax.set_ylabel("count")
    ax.set_title("A  Raw linear intensities (heavy-tailed)", loc="left")

    ax = axes[0, 1]
    ax.hist(obs[rng.choice(obs.size, size=min(300_000, obs.size), replace=False)],
            bins=60, color="#3b6ea5")
    ax.set_xlabel("log2 intensity")
    ax.set_ylabel("count")
    ax.set_title("B  log2 intensities (approximately symmetric)", loc="left")

    # Panel C: per-sample log2 distributions before normalisation, by batch
    ax = axes[1, 0]
    order = sorted(meta_tv["data_source"].unique())
    for i, b in enumerate(order):
        m = (meta_tv["data_source"] == b).to_numpy()
        ax.scatter(np.full(m.sum(), i) + rng.normal(0, 0.08, m.sum()),
                   samp_med_tv[m], s=3, alpha=0.35, color="#c1666b", rasterized=True)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30)
    ax.set_ylabel("per-sample median log2")
    ax.set_title("C  Loading offsets before normalisation", loc="left")

    # Panel D: after normalisation the per-sample medians collapse
    ax = axes[1, 1]
    with np.errstate(all="ignore"):
        med_after = np.nanmedian(norm_tv, axis=1)
    ax.hist(samp_med_tv, bins=60, alpha=0.7, label="before", color="#c1666b")
    ax.hist(med_after, bins=60, alpha=0.9, label="after", color="#3b6ea5")
    ax.set_xlabel("per-sample median log2")
    ax.set_ylabel("number of samples")
    ax.set_title("D  Effect of median normalisation", loc="left")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURES / "eda_intensity_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "eda_intensity_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  saved figures/eda_intensity_distribution.png")


if __name__ == "__main__":
    main()
