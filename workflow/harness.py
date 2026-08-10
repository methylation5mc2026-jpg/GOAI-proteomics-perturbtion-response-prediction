"""Offline re-implementation of the GOAI Track-3 5-dimensional scoring suite.

This module is the single source of truth for *how a model is scored* in this
project.  Nothing here touches the filesystem: every function is pure, so the
same code paths are used by the trivial baselines, by GBDT/deep models, and by
the unit tests in ``11_test_harness.py``.

Handbook provenance
-------------------
The official weight table lives in
``converted/_pdf/赛道三参赛手册.pdf.md`` (the scoring block around lines
327-345).  The PDF->markdown conversion dropped most of the Chinese prose from
those table cells, but the machine-readable parts (weights, split names and the
metric formulae) survived intact and are reproduced verbatim below:

===========================================================  ======  =========================================
Module                                                       Weight  Formula / split
===========================================================  ======  =========================================
Absolute abundance ``corr / R^2``                              20%   PCC and R^2 on log2 abundance
Fold-change response ``FC``                                   25%   ``Delta_pred = y_hat_treat - y_control``
                                                                    ``Delta_true = y_treat  - y_control``
                                                                    ``PCC(Delta_pred, Delta_true)``
S1 residual (novel chemical)                                  20%   ``test_chem_only``; ``mu_ctx`` from train
                                                                    ``PCC(D_pred - mu_ctx, D_true - mu_ctx)``
S2 residual (novel strain)                                    20%   ``test_strain_only``; ``mu_drug`` from train
                                                                    ``PCC(D_pred - mu_drug, D_true - mu_drug)``
S3 + Time                                                     10%   ``test_both`` + ``test_time``, FC + residual
DEP classification                                             5%   ``|Delta_true| > 1``; precision, Recall@K,
                                                                    F1, AUPRC
===========================================================  ======  =========================================

The weights sum to exactly 100%.  The orchestrator's task description splits
the 10% ``S3 + Time`` row into ``S3 = 5%`` and ``Time = 5%``; that is the
convention used here and it reproduces the handbook row total.

Reconstructed vs. specified
---------------------------
Two things are pinned by the handbook and are **not** free parameters: the
module weights, and the metric formulae (which baseline vector is subtracted on
which split).  One thing the handbook does **not** state is the *aggregation
convention* -- whether a PCC is computed by pooling all sample x protein cells
into one vector, or per-sample then averaged, or per-protein then averaged.
Rather than silently guess, this module

* computes **all three** conventions for every correlation, and
* selects the primary one through the explicit, editable :data:`SCORE_SPEC`.

``SCORE_SPEC`` is version-stamped and echoed into every results JSON, so if the
official scorer is later clarified, one edit re-scores every model on record and
the change is visible in the artefacts.  ``score_sensitivity`` re-scores under
all conventions so that model *ranking* stability can be checked independently
of the convention chosen.

Numerical policy
----------------
* Missingness is ubiquitous (~29% of ``Delta`` cells are undefined because
  either the treated sample or its matched control was below detection).  Every
  statistic is therefore computed on the finite intersection mask of the two
  vectors being compared; cells are never imputed and never silently coerced.
* Accumulation is float64 even though the matrices are float32: correlation via
  raw sums of squares loses precision badly in float32 at n ~ 10^7.
* A correlation over a slice with fewer than :data:`MIN_N` finite pairs, or with
  zero variance in either vector, is **NaN**, not 0.0.  NaN means "undefined",
  which is information; it is floored to 0.0 only at the final score-assembly
  step (see :func:`_clip01`) and the count of undefined slices is always
  reported alongside.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

__all__ = [
    "SCORE_SPEC",
    "SPEC_VERSION",
    "DEP_THRESHOLD",
    "MIN_N",
    "masked_pcc",
    "masked_r2",
    "module1_absolute_abundance",
    "module2_fold_change",
    "module3_residual",
    "module4_dep",
    "compute_competition_score",
    "score_sensitivity",
    "flatten_scores",
]

SPEC_VERSION = "1.0.0"

#: Minimum number of finite pairs required before a correlation is reported.
MIN_N = 3

#: A protein is a "differentially expressed protein" (strong responder) when
#: ``|Delta_true| > 1.0`` (log2 units, i.e. a 2-fold change).  Handbook: DEP row.
DEP_THRESHOLD = 1.0

#: Fixed cut-off reported alongside the adaptive (``K = n_positives``) Recall@K.
DEP_FIXED_K = 100

# ---------------------------------------------------------------------------
# Score specification
# ---------------------------------------------------------------------------
#: Module weights, verbatim from the handbook scoring table.  Sum == 1.0.
MODULE_WEIGHTS: dict[str, float] = {
    "m1_abundance": 0.20,
    "m2_fold_change": 0.25,
    "m3_s1_chem": 0.20,
    "m3_s2_strain": 0.20,
    "m3_s3_both": 0.05,
    "m3_time": 0.05,
    "m4_dep": 0.05,
}

#: Which split each module is evaluated on.  Local ``val_*`` names mirror the
#: official ``test_*`` names one-to-one (see ``validation_splits.py``).
MODULE_SPLITS: dict[str, tuple[str, ...]] = {
    "m1_abundance": ("all_val",),
    "m2_fold_change": ("all_val",),
    "m3_s1_chem": ("val_chem_only",),
    "m3_s2_strain": ("val_strain_only",),
    "m3_s3_both": ("val_both",),
    "m3_time": ("val_time",),
    "m4_dep": ("all_val",),
}

#: Sub-metric composition of each module score.  Keys index into the metric
#: dict each module returns; values are sub-weights summing to 1.0 per module.
#: RECONSTRUCTED (aggregation convention) -- see the module docstring.
SUBMETRIC_WEIGHTS: dict[str, dict[str, float]] = {
    # Handbook says "PCC R^2"; the task spec says "overall and per-protein".
    # All four combinations are therefore weighted equally.
    "m1_abundance": {
        "pcc_pooled": 0.25,
        "pcc_per_protein_mean": 0.25,
        "r2_pooled": 0.25,
        "r2_per_protein_mean": 0.25,
    },
    # Response-profile fidelity: per-sample is primary, pooled is a check.
    "m2_fold_change": {"pcc_per_sample_mean": 0.75, "pcc_pooled": 0.25},
    "m3_s1_chem": {"resid_pcc_per_sample_mean": 0.75, "resid_pcc_pooled": 0.25},
    "m3_s2_strain": {"resid_pcc_per_sample_mean": 0.75, "resid_pcc_pooled": 0.25},
    # Handbook: "test_both / test_time -> FC + residual".  Both residual
    # baselines are available on these splits, so both are used.
    "m3_s3_both": {
        "pcc_per_sample_mean": 1 / 3,
        "resid_ctx_pcc_per_sample_mean": 1 / 3,
        "resid_drug_pcc_per_sample_mean": 1 / 3,
    },
    "m3_time": {
        "pcc_per_sample_mean": 1 / 3,
        "resid_ctx_pcc_per_sample_mean": 1 / 3,
        "resid_drug_pcc_per_sample_mean": 1 / 3,
    },
    # Handbook DEP row lists precision, Recall@K, F1 and AUPRC.
    "m4_dep": {
        "precision_macro": 0.25,
        "recall_at_k_macro": 0.25,
        "f1_macro": 0.25,
        "auprc_macro": 0.25,
    },
}

SCORE_SPEC: dict[str, Any] = {
    "spec_version": SPEC_VERSION,
    "source": "converted/_pdf/赛道三参赛手册.pdf.md (scoring table, lines ~327-345)",
    "module_weights": MODULE_WEIGHTS,
    "module_splits": MODULE_SPLITS,
    "submetric_weights": SUBMETRIC_WEIGHTS,
    "dep_threshold": DEP_THRESHOLD,
    "dep_fixed_k": DEP_FIXED_K,
    "min_finite_pairs": MIN_N,
    "negative_correlation_policy": "clipped to 0 when forming a score; raw value always reported",
    "undefined_slice_policy": "NaN (never imputed); excluded from the mean and counted in n_undefined",
    "specified_by_handbook": ["module_weights", "module_splits", "metric formulae", "dep_threshold"],
    "reconstructed_locally": ["submetric_weights (aggregation convention)"],
}


def _clip01(x: float) -> float:
    """Map a correlation/R^2 onto ``[0, 1]`` for score assembly.

    A negative PCC means the prediction is anti-correlated with truth, which is
    no more useful than no prediction at all, so it floors at 0.  R^2 is
    unbounded below and floors identically.  NaN (undefined slice) also floors
    to 0 so that a model cannot gain by making a module unscoreable.
    """
    if x is None or not np.isfinite(x):
        return 0.0
    return float(min(max(x, 0.0), 1.0))


# ---------------------------------------------------------------------------
# NaN-safe vectorised correlation / R^2 primitives
# ---------------------------------------------------------------------------
def _moments(a: np.ndarray, b: np.ndarray, axis: int | None):
    """Masked co-moments of ``a`` and ``b`` reduced along ``axis``.

    Only cells where *both* inputs are finite contribute.  Returns
    ``(n, mean_a, mean_b, var_a, var_b, cov, ss_res)`` where the variances are
    population (``/n``) and ``ss_res`` is ``sum((a - b)^2)`` over the mask.
    """
    m = np.isfinite(a) & np.isfinite(b)
    A = np.where(m, a, 0.0).astype(np.float64, copy=False)
    B = np.where(m, b, 0.0).astype(np.float64, copy=False)

    n = m.sum(axis=axis).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        sa, sb = A.sum(axis=axis), B.sum(axis=axis)
        saa, sbb, sab = (A * A).sum(axis=axis), (B * B).sum(axis=axis), (A * B).sum(axis=axis)
        ss_res = ((A - B) ** 2).sum(axis=axis)

        ma, mb = sa / n, sb / n
        var_a = saa / n - ma * ma
        var_b = sbb / n - mb * mb
        cov = sab / n - ma * mb
    return n, ma, mb, var_a, var_b, cov, ss_res


def masked_pcc(a: np.ndarray, b: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Pearson correlation of ``a`` and ``b`` ignoring non-finite cells.

    Parameters
    ----------
    a, b : numpy.ndarray
        Arrays of identical shape.  NaN marks "not measured".
    axis : int or None
        ``None`` pools every cell into a single scalar correlation.  ``0``
        reduces over samples (one value per protein); ``1`` reduces over
        proteins (one value per sample).

    Returns
    -------
    float or numpy.ndarray
        NaN wherever fewer than :data:`MIN_N` finite pairs exist or either
        vector has zero variance.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    n, _, _, var_a, var_b, cov, _ = _moments(a, b, axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(var_a * var_b)
    bad = (n < MIN_N) | ~(var_a > 0) | ~(var_b > 0)
    r = np.where(bad, np.nan, r)
    # Guard against float round-off pushing |r| a hair past 1.
    r = np.clip(r, -1.0, 1.0)
    return float(r) if axis is None else r


def masked_r2(y_true: np.ndarray, y_pred: np.ndarray,
              axis: int | None = None) -> np.ndarray | float:
    """Coefficient of determination ``1 - SS_res / SS_tot``, NaN-safe.

    ``SS_tot`` is taken about the mean of ``y_true`` *within the same slice and
    mask*, so this is the standard (not the "correlation-squared") R^2 and can
    be negative when the prediction is worse than that slice's mean.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    n, _, _, var_t, _, _, ss_res = _moments(y_true, y_pred, axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        ss_tot = var_t * n
        r2 = 1.0 - ss_res / ss_tot
    bad = (n < MIN_N) | ~(ss_tot > 0)
    r2 = np.where(bad, np.nan, r2)
    return float(r2) if axis is None else r2


def _summarise(vals: np.ndarray, prefix: str) -> dict[str, float]:
    """Reduce a per-axis metric vector to mean/median/undefined-count."""
    vals = np.asarray(vals, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    return {
        f"{prefix}_mean": float(finite.mean()) if finite.size else float("nan"),
        f"{prefix}_median": float(np.median(finite)) if finite.size else float("nan"),
        f"{prefix}_n_defined": int(finite.size),
        f"{prefix}_n_undefined": int(vals.size - finite.size),
    }


# ---------------------------------------------------------------------------
# Module 1 -- absolute abundance (20%)
# ---------------------------------------------------------------------------
def module1_absolute_abundance(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """PCC and R^2 between predicted and true log2 abundance.

    Reported at three aggregations.  ``pooled`` and ``per_sample`` are easy to
    score highly because within any one sample protein abundances span several
    orders of magnitude; ``per_protein`` (across samples, for a fixed protein)
    is the discriminative view and carries half the module weight.
    """
    out: dict[str, float] = {
        "n_cells_finite": int((np.isfinite(y_true) & np.isfinite(y_pred)).sum()),
        "pcc_pooled": masked_pcc(y_true, y_pred, axis=None),
        "r2_pooled": masked_r2(y_true, y_pred, axis=None),
    }
    out.update(_summarise(masked_pcc(y_true, y_pred, axis=1), "pcc_per_sample"))
    out.update(_summarise(masked_pcc(y_true, y_pred, axis=0), "pcc_per_protein"))
    out.update(_summarise(masked_r2(y_true, y_pred, axis=1), "r2_per_sample"))
    out.update(_summarise(masked_r2(y_true, y_pred, axis=0), "r2_per_protein"))
    return out


# ---------------------------------------------------------------------------
# Module 2 -- fold-change response (25%)
# ---------------------------------------------------------------------------
def module2_fold_change(delta_true: np.ndarray, delta_pred: np.ndarray) -> dict[str, float]:
    """PCC between predicted and true perturbation fold-change.

    Both deltas must already have been formed against the *same frozen* control
    anchor (``Delta = y - y_control``), so that the anchor cancels and this
    measures response fidelity rather than baseline calibration.
    """
    out: dict[str, float] = {
        "n_cells_finite": int((np.isfinite(delta_true) & np.isfinite(delta_pred)).sum()),
        "pcc_pooled": masked_pcc(delta_true, delta_pred, axis=None),
        "r2_pooled": masked_r2(delta_true, delta_pred, axis=None),
    }
    out.update(_summarise(masked_pcc(delta_true, delta_pred, axis=1), "pcc_per_sample"))
    out.update(_summarise(masked_pcc(delta_true, delta_pred, axis=0), "pcc_per_protein"))
    return out


# ---------------------------------------------------------------------------
# Module 3 -- residual PCC on the OOD splits (50%)
# ---------------------------------------------------------------------------
def module3_residual(delta_true: np.ndarray, delta_pred: np.ndarray,
                     mu: np.ndarray, prefix: str = "resid") -> dict[str, float]:
    """Residual PCC after removing a train-frozen mean baseline from both sides.

    Subtracting the *same* ``mu`` from prediction and truth removes the part of
    the response that a trivial group-mean model already explains, so the
    residual PCC isolates genuinely novel signal.  A model that merely
    reproduces ``mu`` has zero residual variance and therefore scores NaN, which
    floors to 0 -- this is the intended lower bound, not a bug.

    Parameters
    ----------
    delta_true, delta_pred : numpy.ndarray
        ``(n_samples, n_proteins)`` fold-change matrices.
    mu : numpy.ndarray
        Per-sample baseline, broadcastable to the delta shape.  Rows may be
        all-NaN where the baseline was unresolvable; those cells drop out of the
        mask on both sides symmetrically.
    prefix : str
        Metric-name prefix, used to distinguish ``mu_ctx`` from ``mu_drug``
        residuals on the combined S3/Time splits.
    """
    # No dtype coercion: down-casting a float64 baseline onto float32 deltas
    # leaves a rounding residual, so a model that predicts exactly ``mu`` would
    # show spurious non-zero residual variance instead of the intended NaN.
    mu = np.broadcast_to(np.asarray(mu), delta_true.shape)
    rt = delta_true - mu
    rp = delta_pred - mu
    out: dict[str, float] = {
        f"{prefix}_n_cells_finite": int((np.isfinite(rt) & np.isfinite(rp)).sum()),
        f"{prefix}_pcc_pooled": masked_pcc(rt, rp, axis=None),
    }
    out.update(_summarise(masked_pcc(rt, rp, axis=1), f"{prefix}_pcc_per_sample"))
    out.update(_summarise(masked_pcc(rt, rp, axis=0), f"{prefix}_pcc_per_protein"))
    return out


# ---------------------------------------------------------------------------
# Module 4 -- DEP classification (5%)
# ---------------------------------------------------------------------------
def module4_dep(delta_true: np.ndarray, delta_pred: np.ndarray,
                threshold: float = DEP_THRESHOLD, fixed_k: int = DEP_FIXED_K,
                verbose: bool = False) -> dict[str, float]:
    """Detection of strong responders (``|Delta_true| > threshold``).

    The positive class is "this protein moved by more than ``threshold`` log2
    units in this sample".  The ranking score is ``|Delta_pred|``; the hard
    classifier applies the same ``threshold`` to ``|Delta_pred|`` so that
    precision/recall/F1 need no extra tuned parameter.

    ``Recall@K`` uses the adaptive ``K = n_positives`` per sample (R-precision),
    which is the scale-free choice when the number of true responders varies by
    two orders of magnitude across samples; a fixed-``K`` variant is also
    reported.  Macro statistics average over samples that contain at least one
    positive and at least one negative.
    """
    n_samples = delta_true.shape[0]
    prec = np.full(n_samples, np.nan)
    rec = np.full(n_samples, np.nan)
    f1 = np.full(n_samples, np.nan)
    auprc = np.full(n_samples, np.nan)
    rec_k = np.full(n_samples, np.nan)
    rec_fixed = np.full(n_samples, np.nan)
    n_pos = np.zeros(n_samples, dtype=int)

    tp_tot = fp_tot = fn_tot = 0

    for i in range(n_samples):
        dt, dp = delta_true[i], delta_pred[i]
        m = np.isfinite(dt) & np.isfinite(dp)
        if m.sum() < MIN_N:
            continue
        at, ap = np.abs(dt[m]), np.abs(dp[m])
        y = at > threshold
        n_pos[i] = int(y.sum())
        yhat = ap > threshold

        tp = int((y & yhat).sum())
        fp = int((~y & yhat).sum())
        fn = int((y & ~yhat).sum())
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn

        # Hard-threshold precision / recall / F1 need at least one positive.
        if n_pos[i] > 0:
            rec[i] = tp / n_pos[i]
            if tp + fp > 0:
                prec[i] = tp / (tp + fp)
            denom = 2 * tp + fp + fn
            f1[i] = (2 * tp / denom) if denom > 0 else 0.0

            # Ranking metrics need both classes present.
            if n_pos[i] < y.size:
                auprc[i] = float(average_precision_score(y, ap))
                order = np.argsort(-ap, kind="stable")
                rec_k[i] = y[order[: n_pos[i]]].sum() / n_pos[i]
                k = min(fixed_k, y.size)
                rec_fixed[i] = y[order[:k]].sum() / n_pos[i]

        if verbose and (i + 1) % 500 == 0:
            print(f"      [dep] {i + 1}/{n_samples} samples scored ...", flush=True)

    out: dict[str, float] = {
        "threshold": float(threshold),
        "fixed_k": int(fixed_k),
        "n_samples": int(n_samples),
        "n_positives_total": int(n_pos.sum()),
        "positive_rate_mean": float(np.nanmean(np.where(n_pos > 0, n_pos, np.nan)))
        if (n_pos > 0).any() else float("nan"),
        # Micro (pooled over all cells) hard-threshold statistics.
        "precision_micro": float(tp_tot / (tp_tot + fp_tot)) if (tp_tot + fp_tot) else float("nan"),
        "recall_micro": float(tp_tot / (tp_tot + fn_tot)) if (tp_tot + fn_tot) else float("nan"),
    }
    pm, rm = out["precision_micro"], out["recall_micro"]
    out["f1_micro"] = float(2 * pm * rm / (pm + rm)) if np.isfinite(pm) and np.isfinite(rm) and (pm + rm) > 0 else float("nan")

    for vals, name in [(prec, "precision"), (rec, "recall"), (f1, "f1"),
                       (auprc, "auprc"), (rec_k, "recall_at_k"), (rec_fixed, f"recall_at_{fixed_k}")]:
        s = _summarise(vals, name)
        out[f"{name}_macro"] = s[f"{name}_mean"]
        out[f"{name}_median"] = s[f"{name}_median"]
        out[f"{name}_n_defined"] = s[f"{name}_n_defined"]
    return out


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def _module_score(metrics: dict[str, float], module: str,
                  spec: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Blend a module's sub-metrics into a single ``[0, 1]`` score."""
    sub = spec["submetric_weights"][module]
    parts: dict[str, float] = {}
    total = 0.0
    for key, w in sub.items():
        raw = metrics.get(key, float("nan"))
        val = _clip01(raw)
        parts[key] = raw if raw is not None else float("nan")
        total += w * val
    return float(total), parts


def compute_competition_score(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              delta_true: np.ndarray,
                              delta_pred: np.ndarray,
                              metadata: pd.DataFrame,
                              mu_ctx: np.ndarray | None = None,
                              mu_drug: np.ndarray | None = None,
                              spec: dict[str, Any] | None = None,
                              verbose: bool = True) -> dict[str, Any]:
    """Score a prediction against the full 5-dimensional handbook rubric.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        ``(n, p)`` true and predicted log2 abundance for the evaluated samples.
    delta_true, delta_pred : numpy.ndarray
        ``(n, p)`` fold-changes against the same frozen control anchor.
    metadata : pandas.DataFrame
        ``n`` rows aligned to the matrices, carrying at least ``split_final``
        (values ``val_chem_only`` / ``val_strain_only`` / ``val_both`` /
        ``val_time``).  Row order defines the matrix row order.
    mu_ctx, mu_drug : numpy.ndarray, optional
        ``(n, p)`` train-frozen context-mean and drug-mean ``Delta`` baselines
        for the Module-3 residuals.  Required for the splits that use them; a
        module whose baseline is missing scores 0 and is flagged.
    spec : dict, optional
        Score specification; defaults to :data:`SCORE_SPEC`.
    verbose : bool
        Print per-module progress.

    Returns
    -------
    dict
        ``{"total_score", "module_scores", "module_weights", "modules",
        "coverage", "spec_version", "warnings"}`` where ``modules`` holds the
        full metric dictionary for every module.
    """
    spec = spec or SCORE_SPEC
    n, p = y_true.shape
    for name, arr in [("y_pred", y_pred), ("delta_true", delta_true), ("delta_pred", delta_pred)]:
        if arr.shape != (n, p):
            raise ValueError(f"{name} has shape {arr.shape}, expected {(n, p)}")
    if len(metadata) != n:
        raise ValueError(f"metadata has {len(metadata)} rows, expected {n}")
    if "split_final" not in metadata.columns:
        raise ValueError("metadata must carry a 'split_final' column")

    split = metadata["split_final"].to_numpy()
    warnings: list[str] = []
    modules: dict[str, dict[str, float]] = {}
    module_scores: dict[str, float] = {}
    submetrics: dict[str, dict[str, float]] = {}

    def rows_for(module: str) -> np.ndarray:
        wanted = spec["module_splits"][module]
        if wanted == ("all_val",):
            return np.ones(n, dtype=bool)
        return np.isin(split, wanted)

    # --- Module 1: absolute abundance -------------------------------------
    if verbose:
        print("    [score] module 1: absolute abundance ...", flush=True)
    idx = rows_for("m1_abundance")
    modules["m1_abundance"] = module1_absolute_abundance(y_true[idx], y_pred[idx])
    modules["m1_abundance"]["n_samples"] = int(idx.sum())

    # --- Module 2: fold change -------------------------------------------
    if verbose:
        print("    [score] module 2: fold-change response ...", flush=True)
    idx = rows_for("m2_fold_change")
    modules["m2_fold_change"] = module2_fold_change(delta_true[idx], delta_pred[idx])
    modules["m2_fold_change"]["n_samples"] = int(idx.sum())

    # --- Module 3: residual PCC per OOD split ----------------------------
    resid_plan = [
        ("m3_s1_chem", [("resid", mu_ctx, "mu_ctx")]),
        ("m3_s2_strain", [("resid", mu_drug, "mu_drug")]),
        ("m3_s3_both", [("resid_ctx", mu_ctx, "mu_ctx"), ("resid_drug", mu_drug, "mu_drug")]),
        ("m3_time", [("resid_ctx", mu_ctx, "mu_ctx"), ("resid_drug", mu_drug, "mu_drug")]),
    ]
    for module, baselines in resid_plan:
        idx = rows_for(module)
        if verbose:
            print(f"    [score] module 3: {module} (n={int(idx.sum())}) ...", flush=True)
        met: dict[str, float] = {"n_samples": int(idx.sum())}
        if idx.sum() == 0:
            warnings.append(f"{module}: split empty; module scores 0")
            modules[module] = met
            continue
        # S3/Time also carry a plain fold-change PCC term.
        if "pcc_per_sample_mean" in spec["submetric_weights"][module]:
            met.update(module2_fold_change(delta_true[idx], delta_pred[idx]))
        for prefix, mu, mu_name in baselines:
            if mu is None:
                warnings.append(f"{module}: {mu_name} not supplied; residual term scores 0")
                continue
            met.update(module3_residual(delta_true[idx], delta_pred[idx], mu[idx], prefix=prefix))
        modules[module] = met

    # --- Module 4: DEP classification ------------------------------------
    if verbose:
        print("    [score] module 4: DEP classification ...", flush=True)
    idx = rows_for("m4_dep")
    modules["m4_dep"] = module4_dep(delta_true[idx], delta_pred[idx], verbose=verbose)

    # --- Assemble ---------------------------------------------------------
    total = 0.0
    for module, w in spec["module_weights"].items():
        s, parts = _module_score(modules[module], module, spec)
        module_scores[module] = s
        submetrics[module] = parts
        total += w * s

    return {
        "total_score": float(total),
        "module_scores": module_scores,
        "module_weights": dict(spec["module_weights"]),
        "module_weighted_contributions": {
            k: float(spec["module_weights"][k] * v) for k, v in module_scores.items()
        },
        "primary_submetrics": submetrics,
        "modules": modules,
        "spec_version": spec["spec_version"],
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Aggregation-convention sensitivity
# ---------------------------------------------------------------------------
#: Alternative sub-metric conventions, used to confirm that model ranking does
#: not hinge on the one ambiguity in the handbook.
CONVENTIONS: dict[str, dict[str, dict[str, float]]] = {
    "per_sample_primary": SUBMETRIC_WEIGHTS,
    "pooled_only": {
        "m1_abundance": {"pcc_pooled": 0.5, "r2_pooled": 0.5},
        "m2_fold_change": {"pcc_pooled": 1.0},
        "m3_s1_chem": {"resid_pcc_pooled": 1.0},
        "m3_s2_strain": {"resid_pcc_pooled": 1.0},
        "m3_s3_both": {"pcc_pooled": 1 / 3, "resid_ctx_pcc_pooled": 1 / 3,
                       "resid_drug_pcc_pooled": 1 / 3},
        "m3_time": {"pcc_pooled": 1 / 3, "resid_ctx_pcc_pooled": 1 / 3,
                    "resid_drug_pcc_pooled": 1 / 3},
        "m4_dep": {"precision_micro": 0.25, "recall_at_k_macro": 0.25,
                   "f1_micro": 0.25, "auprc_macro": 0.25},
    },
    "per_sample_only": {
        "m1_abundance": {"pcc_per_sample_mean": 0.5, "r2_per_sample_mean": 0.5},
        "m2_fold_change": {"pcc_per_sample_mean": 1.0},
        "m3_s1_chem": {"resid_pcc_per_sample_mean": 1.0},
        "m3_s2_strain": {"resid_pcc_per_sample_mean": 1.0},
        "m3_s3_both": {"pcc_per_sample_mean": 1 / 3,
                       "resid_ctx_pcc_per_sample_mean": 1 / 3,
                       "resid_drug_pcc_per_sample_mean": 1 / 3},
        "m3_time": {"pcc_per_sample_mean": 1 / 3,
                    "resid_ctx_pcc_per_sample_mean": 1 / 3,
                    "resid_drug_pcc_per_sample_mean": 1 / 3},
        "m4_dep": {"precision_macro": 0.25, "recall_at_k_macro": 0.25,
                   "f1_macro": 0.25, "auprc_macro": 0.25},
    },
    "per_protein_primary": {
        "m1_abundance": {"pcc_per_protein_mean": 0.5, "r2_per_protein_mean": 0.5},
        "m2_fold_change": {"pcc_per_protein_mean": 1.0},
        "m3_s1_chem": {"resid_pcc_per_protein_mean": 1.0},
        "m3_s2_strain": {"resid_pcc_per_protein_mean": 1.0},
        "m3_s3_both": {"pcc_per_protein_mean": 1 / 3,
                       "resid_ctx_pcc_per_protein_mean": 1 / 3,
                       "resid_drug_pcc_per_protein_mean": 1 / 3},
        "m3_time": {"pcc_per_protein_mean": 1 / 3,
                    "resid_ctx_pcc_per_protein_mean": 1 / 3,
                    "resid_drug_pcc_per_protein_mean": 1 / 3},
        "m4_dep": {"precision_macro": 0.25, "recall_at_k_macro": 0.25,
                   "f1_macro": 0.25, "auprc_macro": 0.25},
    },
}


def score_sensitivity(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Re-score an existing result dict under every aggregation convention.

    Uses the already-computed metric dictionaries, so this costs nothing beyond
    a few dict lookups and does not touch the data again.
    """
    out: dict[str, dict[str, float]] = {}
    for cname, subw in CONVENTIONS.items():
        spec = {**SCORE_SPEC, "submetric_weights": subw}
        total = 0.0
        per_module: dict[str, float] = {}
        for module, w in SCORE_SPEC["module_weights"].items():
            s, _ = _module_score(result["modules"][module], module, spec)
            per_module[module] = s
            total += w * s
        out[cname] = {"total_score": float(total), **per_module}
    return out


def flatten_scores(name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a result dict into tidy rows for CSV export.

    Returns one row per (module, metric) pair plus one ``TOTAL`` row.
    """
    rows: list[dict[str, Any]] = [{
        "model": name, "module": "TOTAL", "metric": "total_score",
        "value": result["total_score"], "weight": 1.0,
        "split": "weighted_all", "n_samples": "",
    }]
    for module, metrics in result["modules"].items():
        w = result["module_weights"].get(module, float("nan"))
        splits = "+".join(SCORE_SPEC["module_splits"][module])
        rows.append({
            "model": name, "module": module, "metric": "MODULE_SCORE",
            "value": result["module_scores"][module], "weight": w,
            "split": splits, "n_samples": metrics.get("n_samples", ""),
        })
        for k, v in metrics.items():
            if k == "n_samples":
                continue
            rows.append({
                "model": name, "module": module, "metric": k,
                "value": v, "weight": "", "split": splits,
                "n_samples": metrics.get("n_samples", ""),
            })
    return rows
