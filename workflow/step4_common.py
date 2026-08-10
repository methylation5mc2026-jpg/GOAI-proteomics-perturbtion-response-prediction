"""Shared infrastructure for Step 4 (molecular features, deep learning, stacking).

This module exists so that the RDKit-GBDT script, the deep-learning script and
the stacking script all obtain their data, baselines and scores through exactly
one code path. Any divergence between them would make the stacked score
uninterpretable, so nothing here is duplicated downstream.

Three things are worth calling out because they are load-bearing for the
validity of the Step 4 result:

**1. The molecular block is joined, not rebuilt.** ``data/gbdt_design_train.parquet``
already carries ``perturbation_no_concentration`` per row, so the RDKit columns
can be attached by a lookup on the compound name. This avoids regenerating the
433 MB design matrix (and any risk of it differing from the matrix Step 3 was
trained on). At *prediction* time the compound name cannot be read back off the
design matrix -- novel chemicals are mapped to ``__UNSEEN__`` by the encoder --
so it is always recovered from ``meta`` via the row-to-sample index instead.

**2. Molecular features deliberately survive the chemical regime mask.** The
whole point of adding them is that they transfer to unseen compounds, so they
are attached *after* :func:`features.apply_regime_mask` has blanked the
chemical-keyed group-mean features. They are therefore the only chemical
information a ``chem_novel`` specialist can see.

**3. The inner-dev cohort is what stacking weights may be fitted on.** Fitting
blend weights on ``val_*`` and then reporting the ``val_*`` score would make the
headline number a tuned-on number. :func:`load_inner_context` carves a
regime-matched cohort out of ``train`` only, refits the encoders *and* both
residual baselines on the remaining rows, and relabels its ``split_final`` to
the official names so that the harness computes an objective identical in form
to the reported metric. Weights are then frozen before they ever see ``val_*``.
"""

from __future__ import annotations

from repo_paths import REPO_ROOT

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SESSION = REPO_ROOT
WORKFLOW = SESSION / "workflow"
DATA = SESSION / "data"
RESULTS = SESSION / "results"
FIGURES = SESSION / "figures"
CACHE = DATA / "step4_cache"
MODELS4 = WORKFLOW / "models_step4"

SEED = 42
CHEM_COL = "perturbation_no_concentration"

#: Regime -> the official split label whose feature availability it mirrors.
#: Used to relabel the inner-dev cohort so the harness scores it like the real
#: evaluation set.
REGIME_TO_SPLIT = {
    "chem_novel": "val_chem_only",
    "strain_novel": "val_strain_only",
    "both_novel": "val_both",
    "full": "val_time",
}

T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Molecular feature block
# ---------------------------------------------------------------------------
#: Fingerprint bits are excluded from the tree-model block on purpose: only ~46
#: distinct compounds exist in training, so 320 raw bits would let a tree
#: memorise compound identity rather than learn a structure-activity trend.
#: The compact descriptor + fingerprint-PCA block is what the GBDTs see; the raw
#: bits are reserved for the (heavily regularised) neural network.
def load_mol_features(kind: str = "gbdt") -> pd.DataFrame:
    """Load the per-compound molecular feature table.

    Parameters
    ----------
    kind : {'gbdt', 'full'}
        ``'gbdt'`` returns the descriptor + fingerprint-PCA columns only.
        ``'full'`` additionally returns the raw ECFP4 bit columns.

    Returns
    -------
    pandas.DataFrame
        Indexed by compound name; all columns float32.
    """
    mf = pd.read_parquet(DATA / "step4_mol_features.parquet").set_index(CHEM_COL)
    if kind == "gbdt":
        cols = [c for c in mf.columns if not c.startswith("fp_")]
    elif kind == "full":
        cols = list(mf.columns)
    else:
        raise ValueError(kind)
    return mf[cols].astype("float32")


def mol_block_for_rows(chem_names: np.ndarray, mol: pd.DataFrame) -> pd.DataFrame:
    """Expand a per-compound table to one row per design-matrix row.

    Unknown / non-molecular labels receive the all-zero vector that
    ``22_rdkit_features.py`` assigned them, flagged by ``mol_is_molecule = 0``.
    """
    idx = mol.index.get_indexer(pd.Index(chem_names.astype(str)))
    missing = idx < 0
    if missing.any():
        # Should not happen: the feature table covers every label in both
        # cohorts. Fail loudly rather than silently imputing a chemistry.
        bad = sorted(set(np.asarray(chem_names)[missing].tolist()))[:10]
        raise KeyError(f"{int(missing.sum())} rows have no molecular record, e.g. {bad}")
    return pd.DataFrame(
        mol.to_numpy("float32")[idx], columns=list(mol.columns), copy=False
    )


def attach_mol(X: pd.DataFrame, chem_names: np.ndarray, mol: pd.DataFrame) -> pd.DataFrame:
    """Concatenate the molecular block onto a design matrix, preserving dtypes."""
    blk = mol_block_for_rows(chem_names, mol)
    blk.index = X.index
    return pd.concat([X, blk], axis=1, copy=False)


# ---------------------------------------------------------------------------
# Evaluation context
# ---------------------------------------------------------------------------
def load_context(verbose: bool = True) -> dict:
    """Rebuild the exact Step-3 evaluation objects, plus the Step-4 extras.

    Returns a dict carrying the full train_val cohort (``meta``, ``Y``, ``D``,
    ``C``, ``C_harness``, ``masks``, ``proteins``), the train-frozen residual
    baselines (``mu_ctx``, ``mu_drug``), the abundance fallback (``y_fallback``),
    the group-mean benchmark abundance (``Y_bench``), and a prediction-time
    ``EncoderSet`` fitted on train rows.
    """
    import sys

    sys.path.insert(0, str(WORKFLOW))
    import features as F
    import step3_data as S3
    import validation_splits as VS

    log("loading train_val cohort ...")
    tv = S3.load_train_val(verbose=verbose)
    meta, Y, D, C = tv["meta"], tv["Y"], tv["D"], tv["C"]
    masks, C_harness, proteins = tv["masks"], tv["C_harness"], tv["proteins"]
    train_mask = masks[VS.TRAIN_SPLIT]

    log("verifying split semantics / leakage ...")
    VS.check_no_leakage(meta, masks)

    log("rebuilding train-frozen residual baselines (mu_ctx, mu_drug) ...")
    mu_ctx, mu_drug, base_diag = VS.build_residual_baselines(meta, D, masks)

    log("refitting the benchmark predictor 'per_context_mean_batch' ...")
    Y_bench, _, _ = VS.frozen_delta_baseline(
        meta, Y, train_mask, VS.CTX_LEVELS_BATCH, "abund_ctx_batch"
    )
    y_fallback = _abundance_fallback(meta, Y, train_mask, VS)

    log("fitting prediction-time encoders on train rows ...")
    enc = F.EncoderSet(meta, Y, D, C, train_mask, n_folds=F.N_FOLDS, seed=SEED, verbose=verbose)

    return {
        "meta": meta,
        "Y": Y,
        "D": D,
        "C": C,
        "C_harness": C_harness,
        "masks": masks,
        "proteins": proteins,
        "mu_ctx": mu_ctx,
        "mu_drug": mu_drug,
        "y_fallback": y_fallback,
        "Y_bench": Y_bench,
        "enc": enc,
        "meta_all": tv["meta_all"],
        "M_all": tv["M_all"],
        "baseline_diagnostics": base_diag,
        "F": F,
        "VS": VS,
        "S3": S3,
    }


def _abundance_fallback(meta, Y, train_mask, VS) -> np.ndarray:
    """Train-fitted abundance used where the control anchor is undetected.

    Identical construction to ``16_eval_gbdt.py`` so Module 1 is comparable
    across steps: hierarchical context mean, then per-protein train mean, then
    the global train median for proteins never detected in train.
    """
    Y_ctx_fb, _, _ = VS.frozen_delta_baseline(
        meta, Y, train_mask, VS.CTX_LEVELS, "abund_ctx_fallback"
    )
    with np.errstate(all="ignore"):
        prot_mean_y = np.nanmean(Y[train_mask], axis=0).astype("float32")
        global_med_y = float(np.nanmedian(Y[train_mask]))
    n_never = int((~np.isfinite(prot_mean_y)).sum())
    if n_never:
        log(
            f"  {n_never} proteins never detected in train; abundance fallback is the "
            f"global train median ({global_med_y:.3f})"
        )
        prot_mean_y = np.where(
            np.isfinite(prot_mean_y), prot_mean_y, np.float32(global_med_y)
        ).astype("float32")
    return np.where(
        np.isfinite(Y_ctx_fb), Y_ctx_fb, np.broadcast_to(prot_mean_y, Y.shape)
    ).astype("float32")


def load_inner_context(ctx: dict, verbose: bool = True) -> dict:
    """Carve the regime-matched inner cohort out of ``train`` and refit on it.

    Everything the inner objective touches -- encoders, ``mu_ctx``, ``mu_drug``,
    the benchmark, the abundance fallback -- is refitted on ``inner_fit`` rows
    only, so a model or weight selected against the inner-dev score has never
    seen an inner-dev row. ``val_*`` rows are excluded from both sides entirely.
    """
    F, VS, S3 = ctx["F"], ctx["VS"], ctx["S3"]
    meta, Y, D, C = ctx["meta"], ctx["Y"], ctx["D"], ctx["C"]

    log("carving the regime-matched inner cohort out of train ...")
    fit_mask, devs, info = S3.inner_regime_splits(meta, ctx["masks"], seed=SEED)

    dev_mask = np.zeros(len(meta), dtype=bool)
    regime_of = np.full(len(meta), "", dtype=object)
    for regime, m in devs.items():
        dev_mask |= m
        regime_of[m] = regime
    log(f"  inner_fit n={int(fit_mask.sum())}  inner_dev n={int(dev_mask.sum())}")
    for regime, m in devs.items():
        log(f"    {regime:14s} n={int(m.sum())}")
    assert not (fit_mask & dev_mask).any(), "inner fit/dev overlap"

    inner_masks = {VS.TRAIN_SPLIT: fit_mask, "all_val": dev_mask}
    for regime, split in REGIME_TO_SPLIT.items():
        inner_masks[split] = devs[regime]

    log("refitting mu_ctx / mu_drug on inner_fit rows only ...")
    mu_ctx_i, _, _ = VS.frozen_delta_baseline(
        meta, D, fit_mask, VS.CTX_LEVELS, "inner_mu_ctx"
    )
    mu_drug_i, _, _ = VS.frozen_delta_baseline(
        meta, D, fit_mask, VS.DRUG_LEVELS, "inner_mu_drug"
    )
    log("refitting the benchmark on inner_fit rows only ...")
    Y_bench_i, _, _ = VS.frozen_delta_baseline(
        meta, Y, fit_mask, VS.CTX_LEVELS_BATCH, "inner_abund_ctx_batch"
    )
    y_fb_i = _abundance_fallback(meta, Y, fit_mask, VS)

    log("fitting inner encoders on inner_fit rows only ...")
    enc_i = F.EncoderSet(
        meta, Y, D, C, fit_mask, n_folds=F.N_FOLDS, seed=SEED, verbose=verbose
    )

    # Relabel split_final so the harness scores the inner cohort with the same
    # module/weight structure as the real evaluation set.
    dev_idx = np.flatnonzero(dev_mask)
    meta_dev = meta.iloc[dev_idx].copy().reset_index(drop=True)
    meta_dev["split_final"] = [REGIME_TO_SPLIT[r] for r in regime_of[dev_idx]]

    return {
        "fit_mask": fit_mask,
        "dev_mask": dev_mask,
        "devs": devs,
        "dev_idx": dev_idx,
        "meta_dev": meta_dev,
        "regime_of": regime_of,
        "masks": inner_masks,
        "mu_ctx": mu_ctx_i,
        "mu_drug": mu_drug_i,
        "Y_bench": Y_bench_i,
        "y_fallback": y_fb_i,
        "enc": enc_i,
        "info": info,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(Y_true, Y_pred, D_true, D_pred, meta_eval, mu_ctx, mu_drug) -> dict:
    """Score one ``(abundance, fold-change)`` prediction pair with the harness."""
    import sys

    sys.path.insert(0, str(WORKFLOW))
    import harness as H

    return H.compute_competition_score(
        Y_true, Y_pred, D_true, D_pred, meta_eval, mu_ctx=mu_ctx, mu_drug=mu_drug, verbose=False
    )


def reconstruct(delta: np.ndarray, C: np.ndarray, y_fallback: np.ndarray) -> tuple:
    """Map a fold-change prediction to the ``(y_hat, Delta_hat)`` pair.

    ``y_hat = C + Delta_hat``, with the train-fitted abundance fallback standing
    in wherever the control anchor is undetected. Keeping this coupling is what
    makes the submission self-consistent: the harness accepts abundance and
    fold-change as independent arguments, but a real submission provides only the
    abundance matrix and the organisers derive the fold-change from it, so
    optimising the two separately would not be reproducible at submission time.
    """
    y_hat = C + delta
    gap = ~np.isfinite(y_hat)
    if gap.any():
        y_hat = np.where(gap, y_fallback + np.where(np.isfinite(delta), delta, 0.0), y_hat)
    return y_hat.astype("float32"), delta.astype("float32")


# ---------------------------------------------------------------------------
# Regime routing and molecular-aware prediction
# ---------------------------------------------------------------------------
def regimes_for_samples(enc, meta_src: pd.DataFrame, sample_idx: np.ndarray) -> np.ndarray:
    """Route samples to a regime by entity novelty (identical rule to Step 3).

    Whether a strain / chemical was present in the fit set is knowable at
    submission time for any cohort, so the same routing transfers to the test
    set; a rule keyed on ``split_final`` would not.
    """
    import sys

    sys.path.insert(0, str(WORKFLOW))
    import features as F

    m = meta_src.iloc[sample_idx]
    has_drug = m[CHEM_COL].astype(str).isin(enc.full.seen_chem).to_numpy()
    has_strain = m["Strains"].astype(str).isin(enc.full.seen_strain).to_numpy()
    return F.regime_for_rows(has_drug, has_strain)


def _predict_block(model, lib: str, X: pd.DataFrame, cache: dict) -> np.ndarray:
    """Library-dispatched prediction, reusing per-block frame conversions."""
    import sys

    sys.path.insert(0, str(WORKFLOW))
    import features as F
    import xgboost as xgb

    if lib == "lgb":
        return model.predict(X)
    if lib == "xgb":
        if "dmat" not in cache:
            cache["dmat"] = xgb.DMatrix(X, enable_categorical=True, nthread=24)
        return model.predict(cache["dmat"])
    if "cframe" not in cache:
        out = X.copy()
        for c in F.CAT_FEATURES:
            out[c] = out[c].cat.codes.astype("int32")
        cache["cframe"] = out
    return model.predict(cache["cframe"])


def predict_mol_families(
    models: dict,
    enc,
    sample_idx: np.ndarray,
    meta_src: pd.DataFrame,
    mol: pd.DataFrame | None,
    external: dict | None = None,
    chunk: int = 250,
    label: str = "pred",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Predict every family on one cohort, attaching molecular features.

    ``models`` maps ``family -> {regime: (model, library)}``. Rows are processed
    one regime at a time so each design block is regime-homogeneous and the
    XGBoost/CatBoost frame conversion is shared across families.

    The compound name is taken from ``meta_src`` rather than from the design
    matrix, because the encoder maps novel chemicals to ``__UNSEEN__`` and the
    name would otherwise be unrecoverable for exactly the rows that need it most.
    Pass ``mol=None`` to predict Step-3 style models that expect 37 columns.
    """
    n, p = len(sample_idx), enc.p
    regimes = regimes_for_samples(enc, meta_src, sample_idx)
    counts = {r: int((regimes == r).sum()) for r in set(regimes)}
    log(f"  [{label}] routing: {counts}")

    out = {fam: np.empty((n, p), dtype="float32") for fam in models}
    chem_all = meta_src[CHEM_COL].astype(str).to_numpy()
    t0 = time.time()
    done = 0
    for regime in sorted(set(regimes)):
        pos_all = np.flatnonzero(regimes == regime)
        for s in range(0, len(pos_all), chunk):
            pos = pos_all[s : s + chunk]
            blk = sample_idx[pos]
            X, _, _, row_sample = enc.build_block(blk, external=external)
            if mol is not None:
                X = attach_mol(X, chem_all[row_sample], mol)
            cache: dict = {}
            for fam, by_regime in models.items():
                model, lib = by_regime[regime]
                yp = _predict_block(model, lib, X, cache)
                out[fam][pos] = yp.astype("float32").reshape(len(blk), p)
            del X, cache
            done += len(pos)
            log(f"  [{label}] {done}/{n} samples ({regime}) | {time.time() - t0:.0f}s")
    return out, regimes


# ---------------------------------------------------------------------------
# Prediction cache
# ---------------------------------------------------------------------------
def cache_path(name: str) -> Path:
    """Filesystem location of a cached member prediction matrix."""
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{name}.npy"


def cache_put(name: str, arr: np.ndarray) -> Path:
    """Persist a member's delta matrix so later steps need not recompute it."""
    p = cache_path(name)
    np.save(p, np.ascontiguousarray(arr, dtype="float32"))
    log(f"  cached {name} {arr.shape} -> {p.name}")
    return p


def cache_get(name: str) -> np.ndarray | None:
    """Load a cached member matrix, or ``None`` if it has not been produced."""
    p = cache_path(name)
    return np.load(p) if p.exists() else None


def cache_has(*names: str) -> bool:
    """True only if every requested member matrix is already cached."""
    return all(cache_path(n).exists() for n in names)


def write_json(path: Path, obj) -> None:
    """Write strict JSON, converting numpy scalars/arrays to native types."""

    def enc(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))

    path.write_text(json.dumps(obj, indent=2, default=enc, allow_nan=False), encoding="utf-8")
    log(f"  wrote {path}")
