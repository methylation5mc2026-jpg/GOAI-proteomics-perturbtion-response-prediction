"""Step 3c: score the GBDT models through the harness and predict the test set.

Everything reported here is out-of-sample: the models were fitted on
``split_final == 'train'`` and their hyper-parameters were chosen on a
novel-chemical dev set carved out of ``train``, so the ``val_*`` rows scored
below were never seen in any capacity.

What is computed
----------------
1. **Full 5-module harness score** for the six models (3 libraries x 2 target
   formulations) plus three ensembles, using the *same* ``harness.py`` and the
   *same* train-frozen ``mu_ctx`` / ``mu_drug`` baselines as Step 2 -- so the
   numbers are directly comparable to the 0.4454 benchmark.

2. **Benchmark reproduction.**  ``per_context_mean_batch`` is refitted and
   re-scored here.  Recovering Step 2's 0.445443 to within 1e-9 proves the two
   evaluations are on the same footing; a drift would invalidate the comparison.

3. **Bootstrap confidence intervals** (Step-2 review request).  1,000
   sample-level resamples per split give 95% CIs on each split's per-sample
   metrics, and a *paired* bootstrap on the model-minus-benchmark difference
   answers the question a point estimate cannot: is the improvement larger than
   the sampling noise of a 139-sample split?

4. **Detection-rate sub-cohorts** (Step-2 review request).  Modules 2 and 3 are
   re-scored on proteins with >=50% and >=90% train detection, to show whether
   the gain is uniform or concentrated in well-quantified proteins.

5. **Batch-vs-chemistry attribution.**  Step 2 found that a batch-conditioned
   group mean earns ~0.45 S1 residual PCC while knowing nothing about
   chemistry, because the official ``mu_ctx`` is batch-blind.  Every model is
   therefore *also* scored against a batch-aware ``mu_ctx``, which strips that
   component out.  The gap is the honest measure of how much of the S1/S2
   residual score is biology.

6. **Test predictions** for the 4,226 treated samples of the independent test
   split, from the best model by total score.
"""

from __future__ import annotations

from repo_paths import WORKFLOW_DIR

import json
import platform
import sys
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

# An all-NaN per-sample metric slice is a legitimate "undefined" state that the
# harness already counts and reports; the mean over the defined entries is what
# we want, so the empty-slice warning is noise.
for _msg in ("Mean of empty slice", "All-NaN slice encountered",
             "invalid value encountered", "Degrees of freedom <= 0"):
    warnings.filterwarnings("ignore", message=_msg)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(WORKFLOW_DIR))

import features as F  # noqa: E402
import harness as H  # noqa: E402
import step3_data as S3  # noqa: E402
import validation_splits as VS  # noqa: E402
from common import CHEM_COL, DATA, FIGURES, RESULTS, SEED, WORKFLOW  # noqa: E402

MODELS = WORKFLOW / "models"
#: Family used for the single-model ablation (every row forced through the
#: 'full' specialist), which quantifies what regime routing is worth.
ABLATION_FAMILY = "lgb_delta"
#: Pre-specified (untuned) weight on the GBDT side of the GBDT/benchmark blend.
BLEND_W = 0.5
N_BOOT = 1000
BOOT_SEED = 20260805
#: Train detection-rate thresholds for the protein sub-cohort analysis.
DETECT_CUTS = [0.0, 0.5, 0.9]

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


# ---------------------------------------------------------------------------
# Model loading / prediction
# ---------------------------------------------------------------------------
def _load_one(lib: str, name: str):
    """Load a single booster by library, or ``None`` if the file is absent."""
    import catboost
    import lightgbm as lgb
    import xgboost as xgb

    ext = {"lgb": "txt", "xgb": "json", "cat": "cbm"}[lib]
    path = MODELS / f"{name}.{ext}"
    if not path.exists():
        return None
    if lib == "lgb":
        return lgb.Booster(model_file=str(path))
    if lib == "xgb":
        m = xgb.Booster()
        m.load_model(str(path))
        return m
    m = catboost.CatBoostRegressor()
    m.load_model(str(path))
    return m


def load_models() -> dict[str, dict[str, tuple[object, str, str]]]:
    """Load the regime specialists written by ``15_train_gbdt.py``.

    Returns
    -------
    dict
        ``family -> {regime: (model, library, target_kind)}`` where ``family`` is
        e.g. ``lgb_delta``.  Every family must carry all four regimes, since a
        missing one would silently leave part of the evaluation set unscored.
    """
    families: dict[str, dict[str, tuple[object, str, str]]] = {}
    for target in ("delta", "abs"):
        for lib in ("lgb", "xgb", "cat"):
            fam = f"{lib}_{target}"
            got: dict[str, tuple[object, str, str]] = {}
            for regime in F.REGIMES:
                m = _load_one(lib, f"{fam}__{regime}")
                if m is not None:
                    got[regime] = (m, lib, target)
            if not got:
                continue
            missing = [r for r in F.REGIMES if r not in got]
            if missing:
                raise FileNotFoundError(
                    f"family {fam} is missing regime specialists {missing}; every row must be "
                    "routed to a model trained for its feature-availability regime")
            families[fam] = got
            print(f"  [load] {fam}: {len(got)} regime specialists", flush=True)
    if not families:
        raise FileNotFoundError(f"no trained models found in {MODELS}")
    return families


def regimes_for_samples(enc: F.EncoderSet, meta_src: pd.DataFrame,
                        sample_idx: np.ndarray) -> np.ndarray:
    """Route samples to a regime by *entity novelty*, not by split label.

    Whether a strain / chemical was seen during fitting is knowable at submission
    time for any cohort, so the same routing works unchanged on the test set; a
    rule keyed on ``split_final`` would not generalise beyond this release.
    """
    m = meta_src.iloc[sample_idx]
    has_drug = m[CHEM_COL].astype(str).isin(enc.full.seen_chem).to_numpy()
    has_strain = m["Strains"].astype(str).isin(enc.full.seen_strain).to_numpy()
    return F.regime_for_rows(has_drug, has_strain)


def _cat_frame(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost view of the design matrix: categoricals as integer codes."""
    out = X.copy()
    for c in F.CAT_FEATURES:
        out[c] = out[c].cat.codes.astype("int32")
    return out


def _predict_block(model, lib: str, X: pd.DataFrame, cache: dict) -> np.ndarray:
    """Library-dispatched prediction, reusing per-block conversions."""
    import xgboost as xgb

    if lib == "lgb":
        return model.predict(X)
    if lib == "xgb":
        if "dmat" not in cache:
            cache["dmat"] = xgb.DMatrix(X, enable_categorical=True, nthread=24)
        return model.predict(cache["dmat"])
    if "cframe" not in cache:
        cache["cframe"] = _cat_frame(X)
    return model.predict(cache["cframe"])


def predict_all(families: dict, enc: F.EncoderSet, sample_idx: np.ndarray,
                external: dict | None = None, chunk: int = 250, label: str = "pred",
                force_regime: str | None = None,
                ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Predict every family on one cohort, routing each row to its regime specialist.

    Building the design matrix dominates the cost, so every family scores the same
    block before it is discarded.  Within a block the rows are partitioned by
    regime and each partition is scored by the specialist trained for it.

    No prediction-time masking is needed: a row whose strain is novel *already*
    has its strain-keyed features NaN and ``Strains`` out-of-vocabulary, which is
    exactly the state the specialist was trained on.

    Parameters
    ----------
    force_regime : str, optional
        Route *every* row to this one regime's specialist, ignoring novelty.
        Used to reproduce the single-model ablation (``force_regime='full'``),
        which is the configuration that lost 0.106 of the S2 module score.

    Returns
    -------
    preds : dict
        ``family -> (n, p)`` raw predictions.
    regimes : numpy.ndarray
        ``(n,)`` the regime each sample was routed to.
    """
    meta_src = external["meta"] if external is not None else enc.meta
    n, p = len(sample_idx), enc.p
    regimes = np.full(n, force_regime, dtype=object) if force_regime else \
        regimes_for_samples(enc, meta_src, sample_idx)
    counts = {r: int((regimes == r).sum()) for r in F.REGIMES}
    print(f"  [{label}] routing: {counts}"
          + (f" (forced to '{force_regime}')" if force_regime else ""), flush=True)

    out = {fam: np.empty((n, p), dtype="float32") for fam in families}
    t0 = time.time()
    done = 0
    # Process one regime at a time so every design block is regime-homogeneous:
    # the XGBoost DMatrix / CatBoost frame conversion is then built once per block
    # and shared by all six families, instead of once per (family, regime).
    for regime in F.REGIMES:
        pos_all = np.flatnonzero(regimes == regime)
        if not len(pos_all):
            continue
        for s in range(0, len(pos_all), chunk):
            pos = pos_all[s:s + chunk]
            blk = sample_idx[pos]
            X, _, _, _ = enc.build_block(blk, external=external)
            cache: dict = {}
            for fam, by_regime in families.items():
                model, lib, _ = by_regime[regime]
                yp = _predict_block(model, lib, X, cache)
                out[fam][pos] = yp.astype("float32").reshape(len(blk), p)
            del X, cache
            done += len(pos)
            print(f"  [{label}] {done}/{n} samples ({regime}) | "
                  f"{time.time() - t0:.0f}s", flush=True)
    return out, regimes


def reconstruct(target_kind: str, pred: np.ndarray, C: np.ndarray,
                y_fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map raw model output onto the ``(y_hat, Delta_hat)`` pair the harness needs.

    For the ``delta`` formulation ``y_hat = C + Delta_hat``; where the control
    anchor is undetected, ``y_fallback`` (a train-fitted abundance mean) stands
    in so the cell still contributes to Module 1 instead of dropping out.
    """
    if target_kind == "delta":
        delta_hat = pred
        y_hat = C + pred
        gap = ~np.isfinite(y_hat)
        if gap.any():
            y_hat = np.where(gap, y_fallback + np.where(np.isfinite(pred), pred, 0.0), y_hat)
    elif target_kind == "abs":
        y_hat = pred
        delta_hat = pred - C
    else:
        raise ValueError(target_kind)
    return y_hat.astype("float32"), delta_hat.astype("float32")


def assemble_predictions(families: dict, raw: dict[str, np.ndarray], C: np.ndarray,
                         y_fallback: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reconstruct every model and form the ensembles, for any cohort.

    Used for both the validation and the test cohort so the two cannot drift
    apart.  Ensembles are averaged in ``Delta`` space -- the quantity 80% of the
    rubric scores -- which also keeps the control anchor from being counted once
    per member.  ``ens_all`` therefore averages the *reconstructed* ``Delta`` of
    all six models, not their raw outputs, since the ``abs`` members' raw output
    is an abundance and averaging that with a fold-change would be meaningless.
    """
    def target_of(fam: str) -> str:
        return next(iter(families[fam].values()))[2]

    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fam in families:
        preds[fam] = reconstruct(target_of(fam), raw[fam], C, y_fallback)

    d_names = [fam for fam in families if target_of(fam) == "delta"]
    a_names = [fam for fam in families if target_of(fam) == "abs"]
    if len(d_names) > 1:
        preds["ens_delta"] = reconstruct(
            "delta", np.mean([raw[fam] for fam in d_names], axis=0).astype("float32"),
            C, y_fallback)
    if len(a_names) > 1:
        preds["ens_abs"] = reconstruct(
            "abs", np.mean([raw[fam] for fam in a_names], axis=0).astype("float32"),
            C, y_fallback)
    if len(families) > 1:
        ens_d = np.mean([preds[fam][1] for fam in families], axis=0).astype("float32")
        preds["ens_all"] = reconstruct("delta", ens_d, C, y_fallback)
    return preds


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def boot_ci(vals: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
            ) -> dict[str, float]:
    """Non-parametric bootstrap CI for the mean of a per-sample metric vector.

    NaN entries (undefined slices) are dropped first; resampling is over the
    remaining samples, which is the unit of independence here.
    """
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n": int(v.size)}
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return {"mean": float(v.mean()),
            "ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5)),
            "n": int(v.size)}


def paired_boot(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
                seed: int = BOOT_SEED) -> dict[str, float]:
    """Paired bootstrap on ``mean(a - b)`` over samples where both are defined.

    Pairing removes between-sample variance, which is the dominant noise source
    when two models are scored on the *same* samples; an unpaired comparison of
    two CIs would be far more conservative than the data warrant.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size < 3:
        return {"mean_diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_paired": int(d.size), "p_two_sided": float("nan")}
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    # Two-sided bootstrap p-value: how often the resampled mean crosses zero.
    p = 2.0 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"mean_diff": float(d.mean()),
            "ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5)),
            "n_paired": int(d.size),
            "p_two_sided": float(min(1.0, max(p, 1.0 / n_boot)))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log("=== Step 3c: harness evaluation of the GBDT models ===")
    log(f"numpy {np.__version__} | pandas {pd.__version__} | python {platform.python_version()}")
    log(f"score spec {H.SPEC_VERSION} | bootstrap replicates {N_BOOT} | seed {SEED}")

    # --- Load, rebuild the exact Step-2 evaluation objects -----------------
    tv = S3.load_train_val()
    meta, Y, D, C, masks = tv["meta"], tv["Y"], tv["D"], tv["C"], tv["masks"]
    C_harness, proteins = tv["C_harness"], tv["proteins"]
    train_mask, eval_mask = masks[VS.TRAIN_SPLIT], masks["all_val"]

    log("verifying split semantics and leakage (same assertions as Step 2) ...")
    leak_report = VS.check_no_leakage(meta, masks)

    log("rebuilding the train-frozen residual baselines (mu_ctx, mu_drug) ...")
    mu_ctx, mu_drug, base_diag = VS.build_residual_baselines(meta, D, masks)
    log("rebuilding the batch-aware mu_ctx for the batch-vs-chemistry attribution ...")
    mu_ctx_batch, _, _ = VS.frozen_delta_baseline(meta, D, train_mask,
                                                  VS.CTX_LEVELS_BATCH, "mu_ctx_batch")

    log("refitting the Step-2 benchmark predictor 'per_context_mean_batch' ...")
    Y_bench, _, _ = VS.frozen_delta_baseline(meta, Y, train_mask,
                                             VS.CTX_LEVELS_BATCH, "abund_ctx_batch")
    # Abundance fallback for cells whose control anchor is undetected.
    Y_ctx_fb, _, _ = VS.frozen_delta_baseline(meta, Y, train_mask, VS.CTX_LEVELS,
                                              "abund_ctx_fallback")
    with np.errstate(all="ignore"):
        prot_mean_y = np.nanmean(Y[train_mask], axis=0).astype("float32")
        global_med_y = float(np.nanmedian(Y[train_mask]))
    n_never = int((~np.isfinite(prot_mean_y)).sum())
    if n_never:
        # A protein never detected in any train sample has no per-protein mean.
        # The submission must still carry a finite number for it, so the global
        # train median stands in; these cells cannot be scored against a
        # measured truth anyway wherever the protein is also absent at test time.
        log(f"  {n_never} proteins were never detected in train; their abundance "
            f"fallback is the global train median ({global_med_y:.3f})")
        prot_mean_y = np.where(np.isfinite(prot_mean_y), prot_mean_y,
                               np.float32(global_med_y)).astype("float32")
    y_fallback = np.where(np.isfinite(Y_ctx_fb), Y_ctx_fb,
                          np.broadcast_to(prot_mean_y, Y.shape)).astype("float32")

    log("fitting the prediction-time encoders on train rows ...")
    enc = F.EncoderSet(meta, Y, D, C, train_mask, n_folds=F.N_FOLDS, seed=SEED)

    # --- Predict on the evaluation rows ------------------------------------
    families = load_models()
    eval_idx = np.flatnonzero(eval_mask)
    log(f"predicting {len(families)} model families on {len(eval_idx)} evaluation samples ...")
    raw, eval_regimes = predict_all(families, enc, eval_idx, chunk=250, label="eval")

    # Confirm the novelty-based routing recovers the official split structure.
    route_check = pd.crosstab(meta.loc[eval_mask, "split_final"].to_numpy(), eval_regimes)
    log("regime routing vs split_final:\n" + route_check.to_string())
    expected = {"val_chem_only": "chem_novel", "val_strain_only": "strain_novel",
                "val_both": "both_novel", "val_time": "full"}
    for split, regime in expected.items():
        got = route_check.loc[split].idxmax() if split in route_check.index else None
        n_off = int(route_check.loc[split].sum() - route_check.loc[split].get(regime, 0)) \
            if split in route_check.index else 0
        assert got == regime and n_off == 0, (
            f"{split} routed to {got!r} ({n_off} rows off-regime), expected {regime!r}; "
            "entity-novelty routing must reproduce the split semantics")
    log("  routing check PASSED (entity novelty reproduces the four official splits exactly)")

    # Single-model ablation: route every row through the 'full' specialist, i.e.
    # the configuration a naive pipeline would ship.  Quantifies what the
    # regime-specialist design is worth.
    abl_fam = {k: v for k, v in families.items() if k == ABLATION_FAMILY} \
        if ABLATION_FAMILY in families else {}
    raw_abl = {}
    if abl_fam:
        log(f"ablation: re-predicting '{ABLATION_FAMILY}' with every row forced through "
            f"the 'full' specialist ...")
        raw_abl, _ = predict_all(abl_fam, enc, eval_idx, chunk=250, label="ablation",
                                 force_regime="full")

    meta_eval = meta.loc[eval_mask].reset_index(drop=True)
    Y_ev, D_ev, C_ev = Y[eval_mask], D[eval_mask], C_harness[eval_mask]
    mu_ctx_ev, mu_drug_ev = mu_ctx[eval_mask], mu_drug[eval_mask]
    mu_ctxb_ev = mu_ctx_batch[eval_mask]
    y_fb_ev = y_fallback[eval_mask]

    # --- Assemble the candidate set: models, ensembles, benchmark ----------
    preds = assemble_predictions(families, raw, C_ev, y_fb_ev)
    if raw_abl:
        _, _, tgt = abl_fam[ABLATION_FAMILY]["full"]
        preds[f"{ABLATION_FAMILY}_ABLATION_fullonly"] = reconstruct(
            tgt, raw_abl[ABLATION_FAMILY], C_ev, y_fb_ev)

    bench = np.ascontiguousarray(Y_bench[eval_mask], dtype="float32")
    preds["per_context_mean_batch"] = (bench, bench - C_ev)
    preds["control_anchor"] = (C_ev.copy(), np.zeros_like(C_ev))

    # A fixed equal-weight blend of the GBDT delta ensemble and the group-mean
    # benchmark, averaged in Delta space.  The weight is PRE-SPECIFIED at 0.5 and
    # is not tuned on anything: tuning it would need inner-dev predictions from
    # models that were themselves fitted on the inner dev, and tuning it on
    # val_* would make the reported score a tuned-on number.  Reported as an
    # additional candidate, not as the headline model.
    if "ens_delta" in preds:
        d_bench = bench - C_ev
        d_blend = np.where(np.isfinite(d_bench),
                           BLEND_W * preds["ens_delta"][1] + (1 - BLEND_W) * d_bench,
                           preds["ens_delta"][1]).astype("float32")
        preds["blend_ens_delta_benchmark_w0.5"] = reconstruct("delta", d_blend, C_ev, y_fb_ev)

    # --- Score --------------------------------------------------------------
    results: dict[str, dict] = {}
    sens: dict[str, dict] = {}
    csv_rows: list[dict] = []
    split_rows: list[dict] = []
    per_sample: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for nm, (Yh, Dh) in preds.items():
        log(f"--- scoring '{nm}' ---")
        res = H.compute_competition_score(Y_ev, Yh, D_ev, Dh, meta_eval,
                                         mu_ctx=mu_ctx_ev, mu_drug=mu_drug_ev,
                                         verbose=False)
        results[nm] = res
        sens[nm] = H.score_sensitivity(res)
        csv_rows.extend(H.flatten_scores(nm, res))
        log(f"    total_score = {res['total_score']:.6f} | "
            + " ".join(f"{k.replace('m3_', '').replace('m1_', '').replace('m2_', '')}"
                       f"={v:.3f}" for k, v in res["module_scores"].items()))

        # Per-split detail + the per-sample metric vectors the bootstrap needs.
        per_sample[nm] = {}
        for split in VS.EVAL_SPLITS:
            sm = (meta_eval["split_final"] == split).to_numpy()
            if not sm.any():
                continue
            fc_ps = H.masked_pcc(D_ev[sm], Dh[sm], axis=1)
            rc_ps = H.masked_pcc(D_ev[sm] - mu_ctx_ev[sm], Dh[sm] - mu_ctx_ev[sm], axis=1)
            rd_ps = H.masked_pcc(D_ev[sm] - mu_drug_ev[sm], Dh[sm] - mu_drug_ev[sm], axis=1)
            rb_ps = H.masked_pcc(D_ev[sm] - mu_ctxb_ev[sm], Dh[sm] - mu_ctxb_ev[sm], axis=1)
            ab_ps = H.masked_pcc(Y_ev[sm], Yh[sm], axis=1)
            per_sample[nm][split] = {"fc": fc_ps, "resid_ctx": rc_ps, "resid_drug": rd_ps,
                                     "resid_ctx_batch_aware": rb_ps, "abs": ab_ps}
            split_rows.append({
                "model": nm, "split": split, "n_samples": int(sm.sum()),
                "abs_pcc_per_sample_mean": float(np.nanmean(ab_ps)),
                "fc_pcc_per_sample_mean": float(np.nanmean(fc_ps)),
                "fc_pcc_pooled": H.masked_pcc(D_ev[sm], Dh[sm]),
                "resid_ctx_pcc_per_sample_mean": float(np.nanmean(rc_ps)),
                "resid_drug_pcc_per_sample_mean": float(np.nanmean(rd_ps)),
                "resid_ctx_batch_aware_pcc_per_sample_mean": float(np.nanmean(rb_ps)),
                "batch_attributable_gap": float(np.nanmean(rc_ps) - np.nanmean(rb_ps)),
            })

    # --- Benchmark reproduction check --------------------------------------
    step2 = json.loads((RESULTS / "harness_baseline_scores.json").read_text(encoding="utf-8"))
    ref = float(step2["baseline_totals"]["per_context_mean_batch"])
    got = float(results["per_context_mean_batch"]["total_score"])
    log(f"benchmark reproduction: Step 2 = {ref:.9f}, recomputed here = {got:.9f}, "
        f"|diff| = {abs(got - ref):.2e}")
    assert abs(got - ref) < 1e-9, (
        f"the benchmark predictor no longer reproduces its Step-2 score "
        f"({got:.9f} vs {ref:.9f}); the two evaluations are not comparable")

    # The ablation is a diagnostic, not a candidate; excluding it (and the two
    # reference predictors) keeps "best model" meaning the best shippable model.
    excluded = {"per_context_mean_batch", "control_anchor",
                f"{ABLATION_FAMILY}_ABLATION_fullonly"}
    best = max((nm for nm in preds if nm not in excluded),
               key=lambda k: results[k]["total_score"])
    log(f"best model: '{best}' total_score = {results[best]['total_score']:.6f} "
        f"vs benchmark {got:.6f} (delta = {results[best]['total_score'] - got:+.6f})")

    # --- Bootstrap CIs and paired comparisons ------------------------------
    log(f"bootstrapping ({N_BOOT} replicates) per-split metrics and paired differences ...")
    boot_rows: list[dict] = []
    metric_labels = {"fc": "fold_change_pcc", "resid_ctx": "residual_pcc_vs_mu_ctx",
                     "resid_drug": "residual_pcc_vs_mu_drug",
                     "resid_ctx_batch_aware": "residual_pcc_vs_batch_aware_mu_ctx",
                     "abs": "abundance_pcc"}
    for nm in preds:
        for split, mets in per_sample[nm].items():
            for key, vals in mets.items():
                ci = boot_ci(vals)
                row = {"model": nm, "split": split, "metric": metric_labels[key], **ci}
                if nm != "per_context_mean_batch":
                    row.update({f"vs_benchmark_{k}": v for k, v in paired_boot(
                        vals, per_sample["per_context_mean_batch"][split][key]).items()})
                boot_rows.append(row)
    bdf = pd.DataFrame(boot_rows)
    bdf.to_csv(RESULTS / "gbdt_bootstrap_ci.csv", index=False)
    log(f"wrote results/gbdt_bootstrap_ci.csv ({len(bdf)} rows)")

    key = bdf[(bdf["model"] == best) & (bdf["metric"].isin(
        ["fold_change_pcc", "residual_pcc_vs_mu_ctx"]))]
    print(key[["split", "metric", "mean", "ci_lo", "ci_hi",
               "vs_benchmark_mean_diff", "vs_benchmark_ci_lo", "vs_benchmark_ci_hi",
               "vs_benchmark_p_two_sided"]].round(4).to_string(index=False), flush=True)

    # --- Detection-rate sub-cohorts ----------------------------------------
    log("re-scoring Modules 2/3 on protein detection-rate sub-cohorts ...")
    detect = np.isfinite(Y[train_mask]).mean(axis=0)
    sub_rows: list[dict] = []
    for cut in DETECT_CUTS:
        pm = detect >= cut
        if pm.sum() < 50:
            continue
        for nm in [best, "per_context_mean_batch", "control_anchor"]:
            Yh, Dh = preds[nm]
            for split in VS.EVAL_SPLITS:
                sm = (meta_eval["split_final"] == split).to_numpy()
                if not sm.any():
                    continue
                dt, dp = D_ev[np.ix_(sm, pm)], Dh[np.ix_(sm, pm)]
                mu = mu_ctx_ev[np.ix_(sm, pm)]
                sub_rows.append({
                    "model": nm, "detect_rate_min": cut, "n_proteins": int(pm.sum()),
                    "split": split, "n_samples": int(sm.sum()),
                    "fc_pcc_per_sample_mean": float(np.nanmean(H.masked_pcc(dt, dp, axis=1))),
                    "resid_ctx_pcc_per_sample_mean": float(
                        np.nanmean(H.masked_pcc(dt - mu, dp - mu, axis=1))),
                })
    sdf = pd.DataFrame(sub_rows)
    sdf.to_csv(RESULTS / "gbdt_detection_subcohorts.csv", index=False)
    log(f"wrote results/gbdt_detection_subcohorts.csv ({len(sdf)} rows)")
    print(sdf[sdf["model"] == best].pivot(index="split", columns="detect_rate_min",
                                         values="fc_pcc_per_sample_mean").round(4).to_string(),
          flush=True)

    # --- Exports ------------------------------------------------------------
    pd.DataFrame(csv_rows).to_csv(RESULTS / "gbdt_validation_report.csv", index=False)
    spl = pd.DataFrame(split_rows)
    spl.to_csv(RESULTS / "gbdt_split_metrics.csv", index=False)
    log(f"wrote results/gbdt_validation_report.csv, results/gbdt_split_metrics.csv")

    train_report = json.loads((RESULTS / "gbdt_training_report.json").read_text(encoding="utf-8")) \
        if (RESULTS / "gbdt_training_report.json").exists() else {}

    gate = {
        "benchmark_total_score": got,
        "benchmark_name": "per_context_mean_batch",
        "benchmark_reproduces_step2": True,
        "best_model": best,
        "best_total_score": float(results[best]["total_score"]),
        "beats_benchmark": bool(results[best]["total_score"] > got),
        "margin": float(results[best]["total_score"] - got),
        "per_split_residual_floor_step2": step2["residual_metric_confounds"]["null_floor"][
            "control_anchor_floor_by_split"],
    }
    floors = gate["per_split_residual_floor_step2"]
    bs = spl[spl["model"] == best].set_index("split")
    gate["clears_residual_floor_by_split"] = {
        s: bool(float(bs.loc[s, "resid_ctx_pcc_per_sample_mean"]) > float(v))
        for s, v in floors.items() if s in bs.index}
    log(f"gate: beats_benchmark={gate['beats_benchmark']} (margin {gate['margin']:+.6f}); "
        f"clears residual floor {gate['clears_residual_floor_by_split']}")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step": "3c_gbdt_evaluation", "seed": SEED,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "score_spec": H.SCORE_SPEC,
        "n_eval_samples": int(eval_mask.sum()), "n_proteins": len(proteins),
        "anchor_report": tv["anchor_report"],
        "split_and_leakage_report": leak_report,
        "residual_baseline_diagnostics": base_diag,
        "training_report": train_report,
        "regime_specialists": {
            "rationale": ("the OOD splits delete whole feature families (see "
                          "results/gbdt_feature_availability.json), so one specialist is "
                          "trained per availability regime with exactly those features masked"),
            "routing_rule": ("by entity novelty -- is this sample's strain / chemical present in "
                             "the fit set? -- not by split label, so it transfers to the test set"),
            "regimes": {k: list(v) for k, v in F.REGIMES.items()},
            "chem_dependent_features": F.CHEM_DEPENDENT,
            "strain_dependent_features": F.STRAIN_DEPENDENT,
            "eval_routing_vs_split": jsonable(route_check.to_dict()),
            "ablation": {
                "family": ABLATION_FAMILY,
                "description": ("all rows forced through the 'full' specialist, i.e. the "
                                "single-model pipeline before regime routing"),
                "total_score": float(results[f"{ABLATION_FAMILY}_ABLATION_fullonly"]
                                     ["total_score"]) if raw_abl else None,
                "module_scores": results[f"{ABLATION_FAMILY}_ABLATION_fullonly"]
                ["module_scores"] if raw_abl else None,
                "routed_total_score": float(results[ABLATION_FAMILY]["total_score"])
                if ABLATION_FAMILY in results else None,
            },
        },
        "success_gate": gate,
        "model_totals": {k: float(v["total_score"]) for k, v in results.items()},
        "model_module_scores": {k: v["module_scores"] for k, v in results.items()},
        "models": {k: {"total_score": v["total_score"],
                       "module_scores": v["module_scores"],
                       "module_weights": v["module_weights"],
                       "module_weighted_contributions": v["module_weighted_contributions"],
                       "primary_submetrics": v["primary_submetrics"],
                       "modules": v["modules"], "warnings": v["warnings"]}
                   for k, v in results.items()},
        "aggregation_convention_sensitivity": sens,
        "bootstrap": {"n_replicates": N_BOOT, "seed": BOOT_SEED,
                      "method": ("non-parametric resampling of samples within split; the "
                                 "paired variant resamples the per-sample difference against "
                                 "the benchmark, which removes between-sample variance"),
                      "scope": ("CIs cover the per-sample metric means, which carry 75-100% of "
                                "the weight of Modules 2 and 3; pooled sub-metrics are point "
                                "estimates only"),
                      "rows": jsonable(bdf.to_dict(orient="records"))},
        "detection_subcohorts": jsonable(sdf.to_dict(orient="records")),
        "batch_vs_chemistry_attribution": {
            "method": ("each model's S1/S2 residual PCC is recomputed against a batch-aware "
                       "mu_ctx (data_source added to the grouping key). The official mu_ctx is "
                       "batch-blind, so the gap between the two is the part of the residual "
                       "score explained by plate/instrument offset rather than chemistry."),
            "rows": jsonable(spl.to_dict(orient="records")),
        },
        "split_metrics": jsonable(spl.to_dict(orient="records")),
    }
    (RESULTS / "gbdt_model_scores.json").write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    log("wrote results/gbdt_model_scores.json")

    make_performance_figure(results, got, spl, best)
    make_importance_figure()

    # --- Test-set predictions ----------------------------------------------
    log("=== generating predictions on the independent test set ===")
    te = S3.load_test(proteins, tv["meta_all"], tv["M_all"])
    meta_te, Y_te, D_te, C_te = te["meta"], te["Y"], te["D"], te["C"]
    ext = {"meta": meta_te, "C": C_te, "Y": None, "D": None}

    # Abundance fallback for the test rows, from the same train-frozen tables.
    Y_fb_te, _, _ = project_train_abundance(meta_te, Y, meta, train_mask, VS.CTX_LEVELS,
                                           "abund_fallback_test", prot_mean_y)
    # The benchmark predictor carried onto test, needed only if the blend wins.
    Y_bench_te, _, _ = project_train_abundance(meta_te, Y, meta, train_mask,
                                               VS.CTX_LEVELS_BATCH, "abund_bench_test",
                                               prot_mean_y)

    # Predict every model and assemble through the *same* function used for the
    # validation cohort, then select the winner -- so an ensemble winner is built
    # identically on test and on validation.
    te_idx = np.arange(len(meta_te))
    raw_te, te_regimes = predict_all(families, enc, te_idx, external=ext, chunk=250,
                                     label="test")
    te_route = pd.crosstab(meta_te["split_final"].to_numpy(), te_regimes)
    log("test regime routing vs split_final:\n" + te_route.to_string())
    preds_te = assemble_predictions(families, raw_te, C_te, Y_fb_te)
    if "ens_delta" in preds_te:
        d_bench_te = Y_bench_te - C_te
        d_blend_te = np.where(np.isfinite(d_bench_te),
                              BLEND_W * preds_te["ens_delta"][1]
                              + (1 - BLEND_W) * d_bench_te,
                              preds_te["ens_delta"][1]).astype("float32")
        preds_te["blend_ens_delta_benchmark_w0.5"] = reconstruct(
            "delta", d_blend_te, C_te, Y_fb_te)
    if best not in preds_te:
        raise KeyError(f"best model {best!r} was not reconstructed on the test cohort")
    Yh_te, Dh_te = preds_te[best]
    target_kind = next(iter(families[best].values()))[2] if best in families else \
        ("abs" if best == "ens_abs" else "delta")
    if best.startswith("blend_"):
        target_kind = "delta (blended with the group-mean benchmark, weight 0.5)"
    log(f"test predictions taken from '{best}' (target formulation: {target_kind})")

    # Validate against the required sample IDs before writing anything.
    req = pd.read_parquet(WORKFLOW / "processed_delta_matrix_test.parquet",
                          columns=["sample_ID"])["sample_ID"].astype(str).tolist()
    ids = meta_te["sample_ID"].astype(str).tolist()
    assert ids == req, "test prediction rows are not aligned with the required sample_ID order"
    assert np.isfinite(Yh_te).all(), (
        f"{int((~np.isfinite(Yh_te)).sum())} non-finite predicted abundances; the submitted "
        "abundance matrix must be complete")
    # Delta is a derived quantity, not the submitted matrix.  Under the 'abs'
    # formulation it is undefined wherever the test control anchor is undetected,
    # which is exactly where Delta_true is undefined too, so those cells are
    # unscoreable rather than missing.  Report the coverage instead of asserting.
    delta_cov = float(np.isfinite(Dh_te).mean())
    log(f"test predictions: {Yh_te.shape[0]} samples x {Yh_te.shape[1]} proteins, "
        f"abundance 100% finite, derived Delta defined on {100 * delta_cov:.2f}% of cells; "
        f"sample_ID order verified against the delta matrix")

    out = pd.DataFrame(Yh_te, columns=proteins)
    out.insert(0, "sample_ID", ids)
    out.to_csv(RESULTS / "gbdt_test_predictions.csv", index=False, float_format="%.5f")
    log(f"wrote results/gbdt_test_predictions.csv "
        f"({(RESULTS / 'gbdt_test_predictions.csv').stat().st_size / 1e6:.0f} MB)")
    out.to_parquet(DATA / "gbdt_test_predictions.parquet", index=False, compression="snappy")
    dl = pd.DataFrame(Dh_te, columns=proteins)
    dl.insert(0, "sample_ID", ids)
    dl.to_parquet(DATA / "gbdt_test_delta_predictions.parquet", index=False,
                  compression="snappy")
    log("wrote data/gbdt_test_predictions.parquet, data/gbdt_test_delta_predictions.parquet")

    # Held-out sanity read of the test predictions: the test delta matrix ships
    # with this release, so an indicative (not official) score is available.
    ind = H.module2_fold_change(D_te, Dh_te)
    test_summary = {
        "model": best, "target_formulation": target_kind,
        "n_samples": int(len(ids)), "n_proteins": len(proteins),
        "split_counts": {str(k): int(v) for k, v in meta_te["split_final"].value_counts().items()},
        "y_pred_summary": {"mean": float(Yh_te.mean()), "sd": float(Yh_te.std()),
                           "min": float(Yh_te.min()), "max": float(Yh_te.max())},
        "delta_pred_summary": {"mean": float(np.nanmean(Dh_te)), "sd": float(np.nanstd(Dh_te)),
                               "frac_defined": delta_cov,
                               "frac_abs_gt_1": float(np.nanmean(np.abs(Dh_te) > 1))},
        "indicative_fold_change_pcc_per_sample_mean": float(
            np.nan_to_num(ind["pcc_per_sample_mean"], nan=0.0)),
        "indicative_fold_change_pcc_pooled": float(np.nan_to_num(ind["pcc_pooled"], nan=0.0)),
        "note": ("the released test delta matrix allows an indicative fold-change PCC; it is "
                 "NOT the official score (the official mu_ctx / mu_drug baselines and the "
                 "test-side control matching are held by the organisers)"),
        "anchor_report": te["anchor_report"],
    }
    log(f"indicative test fold-change PCC (per-sample mean) = "
        f"{test_summary['indicative_fold_change_pcc_per_sample_mean']:.4f}")

    payload["test_predictions"] = test_summary
    (RESULTS / "gbdt_model_scores.json").write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")

    print("\n" + "=" * 78)
    print("GBDT TOTAL SCORES (harness spec %s, benchmark to beat = %.4f)" % (H.SPEC_VERSION, got))
    print("=" * 78)
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["total_score"]):
        tag = "  <-- BEST" if k == best else ("  [benchmark]" if k == "per_context_mean_batch" else "")
        print(f"  {k:26s} {v['total_score']:.6f}{tag}")
    print("=" * 78)
    log(f"=== Step 3c complete in {time.time() - T0:.1f}s ===")


def project_train_abundance(meta_te: pd.DataFrame, Y_tv: np.ndarray, meta_tv: pd.DataFrame,
                            train_mask: np.ndarray, levels: list, label: str,
                            prot_mean_y: np.ndarray | None = None):
    """Apply a train-frozen group-mean abundance predictor to the test metadata.

    ``VS.frozen_delta_baseline`` fits group means on masked rows of a single
    frame, so the two cohorts are stacked with the test half of the value matrix
    blanked: the fit therefore sees train rows only, while the lookup resolves
    every test row.  Used twice -- with ``CTX_LEVELS`` for the Module-1 fallback
    and with ``CTX_LEVELS_BATCH`` to carry the benchmark predictor onto test.

    Parameters
    ----------
    prot_mean_y : numpy.ndarray, optional
        Per-protein train mean used where even the global level is undefined, so
        the returned matrix is finite everywhere.  Omit to leave NaN.
    """
    stacked = pd.concat([meta_tv, meta_te], ignore_index=True)
    Y_stacked = np.concatenate([Y_tv, np.full((len(meta_te), Y_tv.shape[1]), np.nan,
                                              dtype="float32")], axis=0)
    tm = np.concatenate([train_mask, np.zeros(len(meta_te), dtype=bool)])
    mu, lvl, info = VS.frozen_delta_baseline(stacked, Y_stacked, tm, levels, label)
    mu_te = mu[len(meta_tv):]
    if prot_mean_y is not None:
        mu_te = np.where(np.isfinite(mu_te), mu_te,
                         np.broadcast_to(prot_mean_y, mu_te.shape)).astype("float32")
    return mu_te.astype("float32"), lvl[len(meta_tv):], info


def make_performance_figure(results: dict, bench: float, spl: pd.DataFrame,
                            best: str) -> None:
    """Three-panel comparison: module scores, totals, and per-split residual PCC."""
    order = sorted(results, key=lambda k: -results[k]["total_score"])
    mods = list(H.MODULE_WEIGHTS)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4),
                             gridspec_kw={"width_ratios": [2.4, 1.15, 1.5]})

    ax = axes[0]
    show = order[:5]
    w = 0.8 / len(show)
    xs = np.arange(len(mods))
    for j, nm in enumerate(show):
        ax.bar(xs + j * w - 0.4 + w / 2,
               [results[nm]["module_scores"][m] for m in mods], w, label=nm)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{m.split('_', 1)[1]}\n{m.split('_')[0]} "
                        f"({H.MODULE_WEIGHTS[m]:.0%})" for m in mods],
                       fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("module score (0-1)")
    ax.set_title("Per-module score, top 5 predictors", fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False, ncol=2)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", lw=0.3, alpha=0.5)

    ax = axes[1]
    tot = [results[n]["total_score"] for n in order][::-1]
    nms = order[::-1]
    colors = ["#C44E52" if n == best else "#8C8C8C" if n in
              ("per_context_mean_batch", "control_anchor") else "#4C72B0" for n in nms]
    ax.barh(nms, tot, color=colors)
    ax.axvline(bench, color="k", ls="--", lw=0.9)
    ax.text(bench, len(nms) - 0.4, f" benchmark {bench:.4f}", fontsize=6.5, va="top")
    for i, v in enumerate(tot):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=6.5)
    ax.set_xlabel("weighted total score")
    ax.set_title("Total competition score", fontsize=9.5)
    ax.set_xlim(0, max(tot) * 1.22)
    ax.tick_params(labelsize=6.8)
    ax.grid(axis="x", lw=0.3, alpha=0.5)

    ax = axes[2]
    sub = spl[spl["model"].isin([best, "per_context_mean_batch", "control_anchor"])]
    splits = list(VS.EVAL_SPLITS)
    w = 0.8 / 3
    for j, (nm, lbl) in enumerate([(best, best),
                                   ("per_context_mean_batch", "benchmark"),
                                   ("control_anchor", "null (Delta=0)")]):
        s = sub[sub["model"] == nm].set_index("split")
        vals = [float(s.loc[sp, "resid_ctx_pcc_per_sample_mean"]) if sp in s.index else np.nan
                for sp in splits]
        ax.bar(np.arange(len(splits)) + j * w - 0.4 + w / 2, vals, w, label=lbl)
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(splits, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("residual PCC vs mu_ctx\n(per-sample mean)")
    ax.set_title("Module-3 residual by OOD split", fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(axis="y", lw=0.3, alpha=0.5)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"gbdt_performance_comparison.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  [fig] figures/gbdt_performance_comparison.png|pdf", flush=True)


def make_importance_figure() -> None:
    """Feature importance, annotated by feature family (batch vs biology)."""
    path = RESULTS / "gbdt_feature_importance.csv"
    if not path.exists():
        print("  [fig] skipping importance figure: no importance CSV", flush=True)
        return
    imp = pd.read_csv(path)

    family = {}
    for f in F.FEATURE_NAMES:
        if f in ("d_batch", "d_batch_time", "d_plate", "d_instr", "data_source",
                 "instrument", "Yeast_cell_plate", "well_row", "well_col",
                 "ctrl_median", "ctrl_detect"):
            family[f] = "batch / technical"
        elif f in ("d_drug", "d_drug_time", "d_drug_strain",
                   "perturbation_no_concentration", "pert_id"):
            family[f] = "chemical"
        elif f in ("c_ctrl", "c_ctrl_centered"):
            family[f] = "control anchor"
        elif f.startswith("prot_"):
            family[f] = "protein statistic"
        else:
            family[f] = "biological context"
    imp["family"] = imp["feature"].map(family).fillna("biological context")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    palette = {"batch / technical": "#C44E52", "chemical": "#55A868",
               "control anchor": "#4C72B0", "protein statistic": "#8172B2",
               "biological context": "#CCB974"}

    has_regime = "regime" in imp.columns
    ax = axes[0]
    ref = imp[(imp["model"] == ABLATION_FAMILY)]
    if has_regime:
        ref = ref[ref["regime"] == "chem_novel"]
    if ref.empty:
        ref = imp[imp["model"] == imp["model"].iloc[0]]
    lbl = f"{ref['model'].iloc[0]}" + (f" / {ref['regime'].iloc[0]}" if has_regime else "")
    ref = ref.sort_values("gain_share", ascending=True).tail(22)
    ax.barh(ref["feature"], ref["gain_share"],
            color=[palette[f] for f in ref["family"]])
    ax.set_xlabel("share of total split gain")
    ax.set_title(f"Feature importance -- {lbl} (top 22)", fontsize=9.5)
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(handles, list(palette), fontsize=7, frameon=False, loc="lower right")

    # Right panel: how the gain redistributes across regimes.  This is the direct
    # evidence that each specialist compensates for what its regime cannot see.
    ax = axes[1]
    key = ["regime"] if has_regime else ["model"]
    sub = imp[imp["model"] == ABLATION_FAMILY] if has_regime else imp
    if sub.empty:
        sub = imp
    fam = sub.groupby(key + ["family"])["gain_share"].sum().unstack(fill_value=0.0)
    if has_regime:
        fam = fam.reindex([r for r in F.REGIMES if r in fam.index])
    fam = fam[[c for c in palette if c in fam.columns]]
    bottom = np.zeros(len(fam))
    xs = np.arange(len(fam))
    for c in fam.columns:
        ax.bar(xs, fam[c].to_numpy(), 0.7, bottom=bottom, label=c, color=palette[c])
        bottom += fam[c].to_numpy()
    ax.set_ylabel("share of total split gain")
    ax.set_title(("Gain by feature family, per availability regime\n"
                  f"({ABLATION_FAMILY} specialists)") if has_regime
                 else "Gain by feature family, per model", fontsize=9.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(fam.index, rotation=25, ha="right", fontsize=7.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"gbdt_feature_importance.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [fig] figures/gbdt_feature_importance.png|pdf", flush=True)


if __name__ == "__main__":
    main()
