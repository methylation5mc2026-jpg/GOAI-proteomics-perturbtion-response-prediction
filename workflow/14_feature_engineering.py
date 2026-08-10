"""Step 3a: build and audit the tabular design matrix for the GBDT baselines.

What this script produces
-------------------------
``data/gbdt_design_train.parquet``
    The long-format training design matrix: one row per ``(train sample,
    protein)`` cell with a finite ``Delta``, sub-sampled at
    :data:`SUBSAMPLE_FRAC`, carrying both targets (``y_abs``, ``y_delta``) and
    the 36 features defined in ``features.py``.
``results/gbdt_feature_audit.json``
    Anchor identity check, out-of-fold self-contribution audit, categorical
    vocabularies, per-feature missingness, and the leakage assertions.
``results/gbdt_feature_coverage.csv``
    Per-split fraction of defined cells for every group-mean feature -- the
    table that explains why the split-specific scores differ.
``figures/gbdt_feature_coverage.png``
    Heat-map of the same.

Why the audits matter
---------------------
The success criterion is "feature matrix generated reproducibly *without data
leakage*".  Three concrete leakage channels exist here and each gets an explicit,
failing-if-violated check:

1. *Statistics fitted on evaluation rows.*  Refitting every encoder with the
   evaluation rows blanked must leave the served features bit-identical
   (``leak_guard_blanked_eval``).
2. *A row's own target inside its own encoding.*  The out-of-fold audit reports
   the mean absolute shift between the OOF value actually served to a fit row and
   the full-fit value; a shift of exactly zero everywhere would mean the OOF
   machinery is not engaged.
3. *Vocabulary leakage.*  Categorical levels are enumerated from fit rows only;
   a level first seen at evaluation time must map to ``__UNSEEN__``, which is
   asserted for the novel strains and chemicals of the OOD splits.
"""

from __future__ import annotations

from repo_paths import WORKFLOW_DIR

import json
import platform
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(WORKFLOW_DIR))

import features as F  # noqa: E402
import step3_data as S3  # noqa: E402
import validation_splits as VS  # noqa: E402
from common import CHEM_COL, DATA, FIGURES, RESULTS, SEED  # noqa: E402

np.random.seed(SEED)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6})

#: Fraction of finite training cells kept in the design matrix.  18.8 M finite
#: train cells is more than a tree ensemble needs; 30% keeps ~5.6 M rows, which
#: still averages ~1,000 observations per protein, and cuts fit time ~3x.
SUBSAMPLE_FRAC = 0.30

T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def jsonable(x):
    """Recursively coerce numpy / non-finite values into JSON-safe values."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


def main() -> None:
    log("=== Step 3a: tabular feature engineering ===")
    log(f"numpy {np.__version__} | pandas {pd.__version__} | python {platform.python_version()}")
    log(f"seed {SEED} | subsample_frac {SUBSAMPLE_FRAC} | n_folds {F.N_FOLDS} "
        f"| min_cell_n {F.MIN_CELL_N}")

    # --- Load -------------------------------------------------------------
    tv = S3.load_train_val()
    meta, Y, D, C, masks = tv["meta"], tv["Y"], tv["D"], tv["C"], tv["masks"]
    proteins = tv["proteins"]
    n, p = D.shape
    log(f"train_val treated: {n} samples x {p} proteins")

    train_mask = masks[VS.TRAIN_SPLIT]
    audit: dict[str, object] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step": "3a_feature_engineering",
        "seed": SEED,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "config": {"subsample_frac": SUBSAMPLE_FRAC, "n_folds": F.N_FOLDS,
                   "min_cell_n": F.MIN_CELL_N,
                   "delta_table_keys": F.DELTA_TABLE_KEYS,
                   "abund_table_keys": F.ABUND_TABLE_KEYS,
                   "cat_features": F.CAT_FEATURES,
                   "n_features": len(F.FEATURE_NAMES)},
        "anchor_report": tv["anchor_report"],
        "n_samples": int(n), "n_proteins": int(p),
        "split_sizes": {k: int(v.sum()) for k, v in masks.items()},
    }

    # --- Fit encoders on train only ---------------------------------------
    log("fitting encoders on train rows (out-of-fold for the fit rows) ...")
    enc = F.EncoderSet(meta, Y, D, C, train_mask, n_folds=F.N_FOLDS, seed=SEED)

    audit["cat_vocabularies"] = {c: {"n_levels_fit": len(v) - 1, "levels": v[:12],
                                     "truncated": len(v) > 12}
                                 for c, v in enc.cat_levels.items()}
    audit["group_table_diagnostics"] = {
        name: {"keys": t.keys, "n_groups": t.n_groups_total,
               "frac_cells_defined": round(t.n_cells_defined, 6)}
        for name, t in {**enc.full.delta, **enc.full.abund}.items()
    }

    # --- Leakage audit 1: out-of-fold self-contribution -------------------
    log("auditing out-of-fold encoding (self-contribution) ...")
    oof = enc.self_contribution_audit(n_samples=60)
    audit["oof_self_contribution"] = oof
    engaged = [k for k, v in oof.items()
               if isinstance(v, dict) and (v["mean_abs_oof_minus_full"] or 0) > 0]
    log(f"  OOF shift is non-zero for {len(engaged)}/{len(F.DELTA_TABLE_KEYS)} tables "
        f"-> the out-of-fold path is engaged")
    for k in list(F.DELTA_TABLE_KEYS)[:5]:
        log(f"    {k:15s} mean|OOF-full| = {oof[k]['mean_abs_oof_minus_full']}")
    assert len(engaged) >= len(F.DELTA_TABLE_KEYS) - 1, (
        "out-of-fold encodings are indistinguishable from full-fit encodings; "
        "the OOF path is not engaged and fit rows would see their own targets")

    # --- Leakage audit 2: refit with evaluation rows blanked --------------
    log("leakage guard: refitting encoders with all evaluation rows blanked ...")
    D_blank, Y_blank, C_blank = D.copy(), Y.copy(), C.copy()
    D_blank[~train_mask] = np.nan
    Y_blank[~train_mask] = np.nan
    C_blank[~train_mask] = np.nan
    enc_blank = F.EncoderSet(meta, Y_blank, D_blank, C_blank, train_mask,
                             n_folds=F.N_FOLDS, seed=SEED, verbose=False)
    same = True
    for tname in list(F.DELTA_TABLE_KEYS) + list(F.ABUND_TABLE_KEYS):
        a = (enc.full.delta.get(tname) or enc.full.abund[tname]).means
        b = (enc_blank.full.delta.get(tname) or enc_blank.full.abund[tname]).means
        if a.shape != b.shape or not np.array_equal(np.nan_to_num(a, nan=-9e9),
                                                    np.nan_to_num(b, nan=-9e9)):
            same = False
            log(f"  MISMATCH in table {tname}")
    for sname in F.PROT_STATS:
        if not np.array_equal(np.nan_to_num(enc.full.prot[sname], nan=-9e9),
                              np.nan_to_num(enc_blank.full.prot[sname], nan=-9e9)):
            same = False
            log(f"  MISMATCH in protein statistic {sname}")
    assert same, "encoders changed when evaluation rows were blanked -> evaluation data leaked"
    log("  leakage guard PASSED (every encoder depends on train rows only)")
    audit["leak_guard_blanked_eval"] = True
    del D_blank, Y_blank, C_blank, enc_blank

    # --- Leakage audit 3: unseen categorical levels ----------------------
    log("checking that OOD entities map to __UNSEEN__ ...")
    unseen_check: dict[str, object] = {}
    for split, col in [("val_chem_only", CHEM_COL), ("val_strain_only", "Strains"),
                       ("val_both", CHEM_COL), ("val_both", "Strains")]:
        m = masks[split]
        lv = set(enc.cat_levels[col][:-1])
        vals = set(meta.loc[m, col].astype(str))
        unseen_check[f"{split}:{col}"] = {
            "n_levels_in_split": len(vals),
            "n_mapped_to_UNSEEN": len(vals - lv),
            "frac_mapped_to_UNSEEN": round(len(vals - lv) / max(1, len(vals)), 6),
        }
    audit["unseen_level_mapping"] = unseen_check
    assert unseen_check["val_chem_only:perturbation_no_concentration"]["frac_mapped_to_UNSEEN"] == 1.0, \
        "val_chem_only chemicals are not all out-of-vocabulary -> vocabulary leakage"
    assert unseen_check["val_strain_only:Strains"]["frac_mapped_to_UNSEEN"] == 1.0, \
        "val_strain_only strains are not all out-of-vocabulary -> vocabulary leakage"
    log("  vocabulary check PASSED (novel chemicals/strains are out-of-vocabulary)")

    # --- Feature coverage per split ---------------------------------------
    log("computing per-split feature coverage ...")
    cov = enc.coverage_report({k: masks[k] for k in
                               [VS.TRAIN_SPLIT, *VS.EVAL_SPLITS]})
    cov.to_csv(RESULTS / "gbdt_feature_coverage.csv", index=False)
    log(f"  wrote results/gbdt_feature_coverage.csv ({len(cov)} rows)")
    piv = cov.pivot(index="feature", columns="split", values="frac_cells_defined")
    print(piv.round(3).to_string(), flush=True)
    audit["feature_coverage_by_split"] = jsonable(
        {c: piv[c].round(6).to_dict() for c in piv.columns})

    make_coverage_figure(piv)

    # --- Build the training design matrix ---------------------------------
    log(f"building the training design matrix (subsample {SUBSAMPLE_FRAC:.0%} of finite cells) ...")
    rng = np.random.default_rng(SEED)
    train_idx = np.flatnonzero(train_mask)

    def cell_mask_fn(block_idx: np.ndarray) -> np.ndarray:
        """Finite-Delta cells, thinned to SUBSAMPLE_FRAC with a per-block seed.

        The seed is derived from the first sample index in the block so the mask
        is reproducible regardless of chunking.
        """
        fin = np.isfinite(D[block_idx])
        r = np.random.default_rng(SEED + int(block_idx[0]))
        return fin & (r.random(fin.shape) < SUBSAMPLE_FRAC)

    X, y_abs, y_delta = F.assemble_training_matrix(
        enc, train_idx, cell_mask_fn, chunk=300, label="design")
    log(f"design matrix: {X.shape[0]:,} rows x {X.shape[1]} features "
        f"({X.memory_usage(deep=True).sum() / 1e9:.2f} GB)")

    # Sanity: both targets finite everywhere (mask required finite Delta, and
    # Delta finite implies Y finite because Delta = Y - C).
    assert np.isfinite(y_delta).all(), "non-finite Delta target survived the mask"
    assert np.isfinite(y_abs).all(), "non-finite abundance target survived the mask"

    miss = (X[F.NUM_FEATURES].isna().mean()).sort_values(ascending=False)
    audit["design_matrix"] = {
        "n_rows": int(X.shape[0]), "n_features": int(X.shape[1]),
        "n_train_samples": int(len(train_idx)),
        "rows_per_protein_mean": round(X.shape[0] / p, 1),
        "target_delta": {"mean": float(y_delta.mean()), "sd": float(y_delta.std()),
                         "q01": float(np.quantile(y_delta, 0.01)),
                         "q99": float(np.quantile(y_delta, 0.99)),
                         "frac_abs_gt_1": float((np.abs(y_delta) > 1).mean())},
        "target_abs": {"mean": float(y_abs.mean()), "sd": float(y_abs.std())},
        "feature_missing_rate": jsonable(miss.round(6).to_dict()),
    }
    log("feature missingness (top 8): "
        + ", ".join(f"{k}={v:.3f}" for k, v in miss.head(8).items()))
    log(f"target Delta: mean={y_delta.mean():.4f} sd={y_delta.std():.4f} "
        f"|Delta|>1 in {100 * (np.abs(y_delta) > 1).mean():.2f}% of rows")

    # --- Persist -----------------------------------------------------------
    out = X.copy()
    out["y_abs"] = y_abs.astype("float32")
    out["y_delta"] = y_delta.astype("float32")
    dest = DATA / "gbdt_design_train.parquet"
    out.to_parquet(dest, index=False, compression="snappy")
    log(f"wrote {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
    del out

    # --- Inner tuning split (definition + assertions only) ----------------
    log("defining the inner chemical-holdout split for hyper-parameter selection ...")
    fit_mask, dev_mask, held = S3.inner_split(meta, masks)
    audit["inner_split"] = {
        "purpose": ("hyper-parameter selection on a novel-chemical holdout carved from "
                    "train only, so no val_* row influences model choice"),
        "holdout_frac_of_train_chemicals": S3.INNER_HOLDOUT_FRAC,
        "held_out_chemicals": held,
        "n_inner_fit": int(fit_mask.sum()), "n_inner_dev": int(dev_mask.sum()),
        "chemical_overlap": 0,
    }

    (RESULTS / "gbdt_feature_audit.json").write_text(
        json.dumps(jsonable(audit), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    log("wrote results/gbdt_feature_audit.json")
    log(f"=== Step 3a complete in {time.time() - T0:.1f}s ===")


def make_coverage_figure(piv: pd.DataFrame) -> None:
    """Heat-map of per-split feature coverage."""
    order = [c for c in [VS.TRAIN_SPLIT, *VS.EVAL_SPLITS] if c in piv.columns]
    M = piv[order].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] < 0.6 else "black")
    ax.set_title("Group-mean feature availability by split\n"
                 "(fraction of cells with a train-fitted value)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="frac cells defined")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"gbdt_feature_coverage.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [fig] figures/gbdt_feature_coverage.png|pdf", flush=True)


if __name__ == "__main__":
    main()
