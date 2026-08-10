"""
Step 4: PCA batch-effect and OOD-split inspection.

Everything learnable is frozen on the 'train' split of train_val:
imputation values, centring, and PCA loadings. All other splits and the test
file are only *projected*, never fitted - this mirrors the competition rule that
reference/normalisation statistics come from training data only.

A variance-attribution table (one-way eta-squared of each PC against each
metadata factor) quantifies how much of each component is batch/instrument
driven versus biology driven. An imputation sensitivity analysis repeats the
whole projection with a left-censored (train 1st-percentile) fill to confirm the
conclusions do not depend on the imputation choice.

Outputs
-------
figures/eda_pca_batch_effects.png
results/pca_scores.csv, results/pca_variance_attribution.csv
results/pca_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import (CHEM_COL, CONTROL_CHEMS, DATA, FIGURES, ID_COL, RESULTS,
                    QC_CHEMS, SEED)

np.random.seed(SEED)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.linewidth": 0.6, "figure.dpi": 110})

DET_THRESHOLD = 0.90   # per-protein detection rate required in BOTH files
N_PC = 10

FACTORS = ["data_source", "instrument", "Strains", "Medium", "Temperature",
           "pert_time", CHEM_COL, "Yeast_cell_plate", "split_final"]


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA eta-squared: fraction of variance explained by a factor.

    Parameters
    ----------
    values : numpy.ndarray
        1-D numeric vector (e.g. one PC's scores).
    groups : numpy.ndarray
        Categorical labels of the same length.

    Returns
    -------
    float
        Ratio of between-group to total sum of squares, in [0, 1].
    """
    ok = np.isfinite(values)
    v, g = values[ok], groups[ok]
    grand = v.mean()
    ss_tot = float(((v - grand) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    ss_between = 0.0
    for lev in pd.unique(g):
        m = g == lev
        ss_between += m.sum() * (v[m].mean() - grand) ** 2
    return float(ss_between / ss_tot)


def build_matrix(fill: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Load normalised log2 matrices, select complete-ish proteins, impute.

    Parameters
    ----------
    fill : {'train_median', 'train_p01'}
        Imputation rule; both are fitted on the train split only.

    Returns
    -------
    x_train, x_all, meta_all, proteins
    """
    tv = pd.read_parquet(DATA.parent / "workflow" / "processed_train_val_proteome.parquet")
    te = pd.read_parquet(DATA.parent / "workflow" / "processed_test_proteome.parquet")
    meta_tv = pd.read_csv(DATA / "meta_train_val_annotated.csv", dtype=str)
    meta_te = pd.read_csv(DATA / "meta_test_annotated.csv", dtype=str)

    proteins_all = [c for c in tv.columns if c != ID_COL]
    m_tv = tv[proteins_all].to_numpy(dtype="float32")
    m_te = te[proteins_all].to_numpy(dtype="float32")

    det_tv = np.isfinite(m_tv).mean(axis=0)
    det_te = np.isfinite(m_te).mean(axis=0)
    keep = (det_tv >= DET_THRESHOLD) & (det_te >= DET_THRESHOLD)
    proteins = [p for p, k in zip(proteins_all, keep) if k]
    print(f"  [pca:{fill}] proteins kept (det>={DET_THRESHOLD} in both): "
          f"{len(proteins)}/{len(proteins_all)}", flush=True)

    m_tv, m_te = m_tv[:, keep], m_te[:, keep]
    meta_all = pd.concat([meta_tv.assign(file="train_val"),
                          meta_te.assign(file="test")], ignore_index=True)
    x_all = np.vstack([m_tv, m_te])
    train_mask = (meta_all["split_final"] == "train").to_numpy() & \
                 (meta_all["file"] == "train_val").to_numpy()

    # Imputation values frozen on the train split
    with np.errstate(all="ignore"):
        if fill == "train_median":
            fill_vals = np.nanmedian(x_all[train_mask], axis=0)
        elif fill == "train_p01":
            fill_vals = np.nanpercentile(x_all[train_mask], 1, axis=0)
        else:
            raise ValueError(f"unknown fill rule {fill!r}")
    fill_vals = np.where(np.isfinite(fill_vals), fill_vals,
                         np.nanmedian(fill_vals[np.isfinite(fill_vals)]))
    n_imp = int((~np.isfinite(x_all)).sum())
    idx = np.where(~np.isfinite(x_all))
    x_all[idx] = fill_vals[idx[1]]
    print(f"  [pca:{fill}] imputed {n_imp} cells "
          f"({100 * n_imp / x_all.size:.2f}%) with train-split {fill}", flush=True)

    return x_all, train_mask, meta_all, proteins


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building matrix (primary imputation: train_median) ...")
    x_all, train_mask, meta_all, proteins = build_matrix("train_median")

    print("[2/4] Fitting PCA on the train split only, projecting everything ...")
    center = x_all[train_mask].mean(axis=0)
    pca = PCA(n_components=N_PC, random_state=SEED, svd_solver="randomized")
    pca.fit(x_all[train_mask] - center)
    scores = pca.transform(x_all - center)
    evr = pca.explained_variance_ratio_
    print(f"  explained variance ratio (train-fitted): "
          f"{np.round(evr, 4).tolist()}")
    print(f"  cumulative: {np.round(np.cumsum(evr), 4).tolist()}")

    sc = pd.DataFrame(scores[:, :N_PC], columns=[f"PC{i+1}" for i in range(N_PC)])
    sc.insert(0, ID_COL, meta_all[ID_COL].to_numpy())
    for c in ["file"] + FACTORS + ["sample_role"]:
        sc[c] = meta_all[c].to_numpy()
    sc.to_csv(RESULTS / "pca_scores.csv", index=False)
    print(f"  saved results/pca_scores.csv ({sc.shape})")

    print("[3/4] Variance attribution (eta-squared per PC per factor) ...")
    rows = []
    for f in FACTORS:
        g = meta_all[f].astype(str).to_numpy()
        row = {"factor": f, "n_levels": int(pd.unique(g).size)}
        for i in range(min(5, N_PC)):
            row[f"eta2_PC{i+1}"] = round(eta_squared(scores[:, i], g), 4)
        rows.append(row)
    attrib = pd.DataFrame(rows).sort_values("eta2_PC1", ascending=False)
    attrib.to_csv(RESULTS / "pca_variance_attribution.csv", index=False)
    print(attrib.to_string(index=False))

    print("\n[4/4] Imputation sensitivity: repeating with left-censored train_p01 fill ...")
    x2, tm2, _, _ = build_matrix("train_p01")
    c2 = x2[tm2].mean(axis=0)
    p2 = PCA(n_components=N_PC, random_state=SEED, svd_solver="randomized")
    p2.fit(x2[tm2] - c2)
    s2 = p2.transform(x2 - c2)
    # Components can swap order / rotate between runs, so compare (a) each PC
    # against its BEST match in the alternative run and (b) the top-4 subspaces
    # as a whole via principal angles. Diagonal-only correlation would wrongly
    # report instability for a simple PC2/PC3 swap.
    k = 4
    cross = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cross[i, j] = abs(np.corrcoef(scores[:, i], s2[:, j])[0, 1])
    sens = {"cross_abs_corr_matrix": np.round(cross, 3).tolist()}
    print("  |corr| matrix (rows=primary PC, cols=left-censored PC):")
    for i in range(k):
        print(f"    PC{i+1}: {np.round(cross[i], 3).tolist()}")
    for i in range(k):
        j = int(np.argmax(cross[i]))
        sens[f"PC{i+1}_best_match"] = {"matches": f"PC{j+1}",
                                       "abs_corr": round(float(cross[i, j]), 4)}
        print(f"  primary PC{i+1} best matches left-censored PC{j+1} "
              f"(|corr|={cross[i, j]:.4f})")

    # Principal angles between the two 4-D score subspaces
    qa, _ = np.linalg.qr(scores[:, :k] - scores[:, :k].mean(0))
    qb, _ = np.linalg.qr(s2[:, :k] - s2[:, :k].mean(0))
    sv = np.linalg.svd(qa.T @ qb, compute_uv=False)
    sv = np.clip(sv, -1, 1)
    angles = np.degrees(np.arccos(sv))
    sens["subspace_principal_angles_deg"] = np.round(angles, 2).tolist()
    sens["mean_subspace_overlap"] = round(float(sv.mean()), 4)
    print(f"  principal angles between top-{k} subspaces (deg): "
          f"{np.round(angles, 2).tolist()}")
    print(f"  mean subspace overlap (cos of angles): {sv.mean():.4f}")
    sens["evr_train_p01"] = np.round(p2.explained_variance_ratio_, 4).tolist()

    _figure(sc, evr, attrib)

    summary = {
        "n_proteins_used": len(proteins),
        "detection_threshold": DET_THRESHOLD,
        "n_samples": int(x_all.shape[0]),
        "pca_fitted_on": "split_final == 'train' of train_val only",
        "explained_variance_ratio": np.round(evr, 5).tolist(),
        "cumulative_variance_ratio": np.round(np.cumsum(evr), 5).tolist(),
        "variance_attribution_eta2": attrib.to_dict(orient="records"),
        "imputation_sensitivity": sens,
    }
    (RESULTS / "pca_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved results/pca_summary.json")


def _figure(sc: pd.DataFrame, evr: np.ndarray, attrib: pd.DataFrame) -> None:
    """Six-panel PCA diagnostic: colour by batch, strain, split, instrument + scree."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(sc))  # avoid draw-order bias
    s = sc.iloc[order]

    def scatter_by(ax, col, title, max_levels=12, legend=True):
        levels = s[col].value_counts().index.tolist()
        shown = levels[:max_levels]
        cmap = plt.get_cmap("tab20" if len(shown) > 10 else "tab10")
        ax.scatter(s.loc[~s[col].isin(shown), "PC1"],
                   s.loc[~s[col].isin(shown), "PC2"],
                   s=2, c="#dddddd", alpha=0.4, rasterized=True,
                   label="other" if len(levels) > max_levels else None)
        for i, lv in enumerate(shown):
            m = s[col] == lv
            ax.scatter(s.loc[m, "PC1"], s.loc[m, "PC2"], s=2.5, alpha=0.6,
                       color=cmap(i % cmap.N), label=str(lv)[:22], rasterized=True)
        ax.set_xlabel(f"PC1 ({100*evr[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100*evr[1]:.1f}%)")
        ax.set_title(title, loc="left")
        if legend:
            ax.legend(frameon=False, markerscale=3, fontsize=6, loc="best", ncol=1)

    scatter_by(axes[0, 0], "data_source", "A  Batch (data_source)")
    scatter_by(axes[0, 1], "Strains", "B  Strain")
    scatter_by(axes[0, 2], "split_final", "C  Official evaluation split")
    scatter_by(axes[1, 0], "instrument", "D  Instrument")

    # Panel E: scree
    ax = axes[1, 1]
    ax.bar(np.arange(1, len(evr) + 1), 100 * evr, color="#3b6ea5", width=0.7)
    ax.plot(np.arange(1, len(evr) + 1), 100 * np.cumsum(evr), "o-",
            color="#c1666b", ms=3, lw=1, label="cumulative")
    ax.set_xlabel("principal component")
    ax.set_ylabel("% variance explained")
    ax.set_title("E  Scree (train-fitted)", loc="left")
    ax.legend(frameon=False)

    # Panel F: eta-squared heat map factor x PC
    ax = axes[1, 2]
    cols = [c for c in attrib.columns if c.startswith("eta2_")]
    mat = attrib.set_index("factor")[cols].to_numpy(dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("eta2_", "") for c in cols])
    ax.set_yticks(range(len(attrib)))
    ax.set_yticklabels([f[:24] for f in attrib["factor"]], fontsize=6)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if mat[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, label="eta-squared")
    ax.set_title("F  Variance attribution per PC", loc="left")

    fig.tight_layout()
    fig.savefig(FIGURES / "eda_pca_batch_effects.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "eda_pca_batch_effects.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  saved figures/eda_pca_batch_effects.png")


if __name__ == "__main__":
    main()
