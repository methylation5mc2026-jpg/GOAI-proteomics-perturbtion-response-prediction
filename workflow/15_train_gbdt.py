"""Step 3b: train the GBDT baselines under both target formulations.

Two formulations, three libraries
--------------------------------
``delta``
    Predict ``Delta_hat`` directly; reconstruct
    ``y_hat = C + Delta_hat`` (falling back to the train-fitted
    batch-context mean abundance where the control anchor is undetected).
``abs``
    Predict ``y_hat`` directly; derive ``Delta_hat = y_hat - C``.

Each is trained with LightGBM, XGBoost and CatBoost, giving six models plus the
equal-weight ensembles that ``16_eval_gbdt.py`` assembles.  Formulation matters
because 80% of the rubric weight sits on ``Delta`` (Modules 2-4) while 20% sits on
absolute abundance, and the two formulations put their modelling capacity in
different places: ``delta`` spends every split on the perturbation response and
inherits a near-perfect baseline for free, whereas ``abs`` must first spend
capacity re-learning the protein's absolute level.

Hyper-parameter selection without touching the reported splits
-------------------------------------------------------------
Selecting on ``val_*`` would make the headline number a tuned-on number.  Instead
a novel-chemical dev set is carved out of ``train`` alone
(``step3_data.inner_split``): :data:`INNER_TUNE_FRAC` of the finite cells of
``inner_fit`` train the candidates, early stopping watches ``inner_dev``, and the
selection objective is the weight-proportional blend of the two dominant modules

``obj = (0.25 * m2_pcc_per_sample + 0.20 * s1_residual_pcc_per_sample) / 0.45``

with the residual baseline ``mu_ctx`` refitted on ``inner_fit`` only.  The winning
configuration is then refitted on all of ``train``.  No ``val_*`` row is read
anywhere in this script.

Outputs
-------
``workflow/models/*.{txt,json,cbm}``   trained boosters
``results/gbdt_tuning_trials.csv``     every candidate and its inner-dev score
``results/gbdt_training_report.json``  chosen config, fit times, train RMSE
"""

from __future__ import annotations

from repo_paths import WORKFLOW_DIR

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(WORKFLOW_DIR))

import harness as H  # noqa: E402
import features as F  # noqa: E402
import step3_data as S3  # noqa: E402
import validation_splits as VS  # noqa: E402
from common import DATA, RESULTS, SEED, WORKFLOW  # noqa: E402

#: Cell-subsample fraction for the *tuning* matrices.  Smaller than the final
#: 30% so eight candidates fit in a few minutes; configuration ranking is stable
#: at this size because the objective averages over hundreds of dev samples.
INNER_TUNE_FRAC = 0.12

#: Dev samples used for the tuning objective (a mean over samples; a few hundred
#: is ample and keeps per-candidate prediction cost low).
N_DEV_SAMPLES = 400

#: Rounds each candidate is trained for during tuning.  Selection then walks the
#: ladder below and scores the *competition* objective at each truncation, rather
#: than early-stopping on squared error: the rubric is a correlation, and the
#: iteration that minimises L2 on held-out cells is not generally the iteration
#: that maximises per-sample PCC.
MAX_ROUNDS = 500
ROUND_LADDER = [50, 100, 200, 350, 500]

#: When refitting on all of train the row count grows ~1/(1 - inner_holdout);
#: scale the tuned iteration count by the same factor rather than early-stopping
#: against data we are about to train on.
REFIT_ROUND_SCALE = 1.20

MODELS = WORKFLOW / "models"
T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Candidate grid
# ---------------------------------------------------------------------------
def candidate_configs() -> list[dict]:
    """LightGBM candidates spanning depth, shrinkage, regularisation and loss.

    Deliberately small and axis-aligned rather than a random search: with a
    single dev set, a large search would mostly fit the dev noise.
    """
    base = {"num_leaves": 63, "learning_rate": 0.06, "min_data_in_leaf": 200,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 1.0, "objective": "regression", "max_depth": -1}
    grid = [
        ("base", {}),
        ("shallow", {"num_leaves": 31, "learning_rate": 0.08}),
        ("deep", {"num_leaves": 127, "min_data_in_leaf": 400}),
        ("deep_slow", {"num_leaves": 127, "learning_rate": 0.03, "min_data_in_leaf": 400}),
        ("strong_l2", {"lambda_l2": 20.0}),
        ("low_ff", {"feature_fraction": 0.55}),
        ("huber", {"objective": "huber", "alpha": 1.0}),
        ("deep_huber", {"num_leaves": 127, "min_data_in_leaf": 400,
                        "objective": "huber", "alpha": 1.0}),
    ]
    out = []
    for name, over in grid:
        cfg = {**base, **over}
        cfg["name"] = name
        out.append(cfg)
    return out


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def build_dev_design(enc: F.EncoderSet, dev_idx: np.ndarray, chunk: int = 200,
                     label: str = "dev") -> pd.DataFrame:
    """Full (unmasked) design matrix for the dev samples, built once and reused.

    Every candidate and every point on :data:`ROUND_LADDER` is scored on this
    same matrix, so the block-building cost is paid a single time instead of
    ``n_candidates x len(ROUND_LADDER)`` times.
    """
    parts = []
    t0 = time.time()
    for s in range(0, len(dev_idx), chunk):
        blk = dev_idx[s:s + chunk]
        X, _, _, _ = enc.build_block(blk)
        parts.append(X)
        print(f"  [{label}] {min(s + chunk, len(dev_idx))}/{len(dev_idx)} samples | "
              f"{time.time() - t0:.0f}s", flush=True)
    return pd.concat(parts, ignore_index=True, copy=False)


def _cat_frame(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost view of the design matrix: categoricals as integer codes.

    Integer codes avoid materialising millions of Python strings; CatBoost treats
    an integer categorical feature exactly as it would a string one.
    """
    out = X.copy()
    for c in F.CAT_FEATURES:
        out[c] = out[c].cat.codes.astype("int32")
    return out


# ---------------------------------------------------------------------------
# Tuning objective
# ---------------------------------------------------------------------------
def regime_objective(regime: str, D_true: np.ndarray, D_pred: np.ndarray,
                     mu_ctx: np.ndarray, mu_drug: np.ndarray) -> dict[str, float]:
    """Selection objective for one availability regime.

    Each regime feeds a specific set of rubric modules, so each is selected on a
    blend weighted the way the rubric weights those modules:

    * ``chem_novel``   -> Module 2 (25%) + ``m3_s1_chem`` (20%, residual vs ``mu_ctx``)
    * ``strain_novel`` -> Module 2 (25%) + ``m3_s2_strain`` (20%, residual vs ``mu_drug``)
    * ``both_novel``   -> Module 2 (25%) + ``m3_s3_both`` (5%, the three-term mean)
    * ``full``         -> Module 2 (25%) + ``m3_time`` (5%, the same three-term mean)

    Negative correlations are floored at 0 exactly as ``harness._clip01`` does, so
    the objective is on the same scale as a module score.
    """
    m2 = H.module2_fold_change(D_true, D_pred)
    rc = H.module3_residual(D_true, D_pred, mu_ctx, prefix="c")
    rd = H.module3_residual(D_true, D_pred, mu_drug, prefix="d")

    def g(x):
        return max(float(np.nan_to_num(x, nan=0.0)), 0.0)

    a = g(m2["pcc_per_sample_mean"])
    b_ctx = g(rc["c_pcc_per_sample_mean"])
    b_drug = g(rd["d_pcc_per_sample_mean"])

    if regime == "chem_novel":
        obj = (0.25 * a + 0.20 * b_ctx) / 0.45
    elif regime == "strain_novel":
        obj = (0.25 * a + 0.20 * b_drug) / 0.45
    else:                                   # both_novel, full -> the S3/Time form
        obj = (0.25 * a + 0.05 * (a + b_ctx + b_drug) / 3.0) / 0.30
    return {"m2_pcc_per_sample": a,
            "resid_pcc_vs_mu_ctx": b_ctx,
            "resid_pcc_vs_mu_drug": b_drug,
            "m2_pcc_pooled": float(np.nan_to_num(m2["pcc_pooled"], nan=0.0)),
            "objective": float(obj)}


# ---------------------------------------------------------------------------
# Library-specific fitting
# ---------------------------------------------------------------------------
def fit_lgb(X: pd.DataFrame, y: np.ndarray, cfg: dict, n_rounds: int,
            valid: tuple | None = None, seed: int = SEED):
    """Fit a LightGBM booster; categoricals are consumed natively."""
    import lightgbm as lgb
    params = {k: v for k, v in cfg.items() if k != "name"}
    params.update({"verbosity": -1, "num_threads": 24, "seed": seed,
                   "deterministic": True, "force_row_wise": True,
                   "cat_smooth": 20, "min_data_per_group": 100,
                   "max_cat_threshold": 64})
    dtrain = lgb.Dataset(X, label=y, categorical_feature=F.CAT_FEATURES,
                         free_raw_data=False)
    cbs = [lgb.log_evaluation(period=200)]
    valid_sets = [lgb.Dataset(valid[0], label=valid[1], reference=dtrain)] \
        if valid is not None else []
    return lgb.train(params, dtrain, num_boost_round=n_rounds,
                     valid_sets=valid_sets, callbacks=cbs)


def fit_xgb(X: pd.DataFrame, y: np.ndarray, cfg: dict, n_rounds: int, seed: int = SEED):
    """Fit an XGBoost booster with native categorical support."""
    import xgboost as xgb
    obj = "reg:pseudohubererror" if cfg.get("objective") == "huber" else "reg:squarederror"
    params = {
        "objective": obj, "tree_method": "hist", "max_bin": 256,
        "max_leaves": int(cfg["num_leaves"]), "grow_policy": "lossguide",
        "max_depth": 0, "eta": float(cfg["learning_rate"]),
        "min_child_weight": float(cfg["min_data_in_leaf"]) / 10.0,
        "colsample_bytree": float(cfg["feature_fraction"]),
        "subsample": float(cfg["bagging_fraction"]),
        "reg_lambda": float(cfg["lambda_l2"]),
        "max_cat_to_onehot": 8, "nthread": 24, "seed": seed,
    }
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True, nthread=24)
    return xgb.train(params, dtrain, num_boost_round=n_rounds,
                     evals=[(dtrain, "train")], verbose_eval=100)


def fit_cat(X: pd.DataFrame, y: np.ndarray, cfg: dict, n_rounds: int, seed: int = SEED):
    """Fit a CatBoost regressor; categoricals passed as integer codes."""
    from catboost import CatBoostRegressor, Pool
    loss = "Huber:delta=1.0" if cfg.get("objective") == "huber" else "RMSE"
    Xc = _cat_frame(X)
    pool = Pool(Xc, label=y, cat_features=F.CAT_FEATURES)
    model = CatBoostRegressor(
        iterations=n_rounds, learning_rate=float(cfg["learning_rate"]),
        depth=min(10, max(4, int(np.log2(cfg["num_leaves"])) + 1)),
        l2_leaf_reg=float(cfg["lambda_l2"]), loss_function=loss,
        rsm=float(cfg["feature_fraction"]), random_seed=seed,
        min_data_in_leaf=int(cfg["min_data_in_leaf"]), border_count=128,
        thread_count=24, verbose=100, allow_writing_files=False,
        bootstrap_type="Bernoulli", subsample=float(cfg["bagging_fraction"]),
    )
    model.fit(pool)
    return model


FITTERS = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}


def importance(model, kind: str) -> pd.Series:
    """Gain-based feature importance, normalised to sum to 1."""
    if kind == "lgb":
        s = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    elif kind == "xgb":
        d = model.get_score(importance_type="total_gain")
        s = pd.Series({f: d.get(f, 0.0) for f in F.FEATURE_NAMES})
    else:
        s = pd.Series(model.get_feature_importance(), index=model.feature_names_)
    return (s / s.sum()) if s.sum() > 0 else s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log("=== Step 3b: GBDT training ===")
    import lightgbm as lgb
    import xgboost as xgb
    import catboost
    log(f"lightgbm {lgb.__version__} | xgboost {xgb.__version__} | "
        f"catboost {catboost.__version__} | python {platform.python_version()}")
    MODELS.mkdir(parents=True, exist_ok=True)

    tv = S3.load_train_val()
    meta, Y, D, C, masks = tv["meta"], tv["Y"], tv["D"], tv["C"], tv["masks"]
    train_mask = masks[VS.TRAIN_SPLIT]

    report: dict[str, object] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step": "3b_gbdt_training", "seed": SEED,
        "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                     "xgboost": xgb.__version__, "catboost": catboost.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "config": {"inner_tune_frac": INNER_TUNE_FRAC, "n_dev_samples": N_DEV_SAMPLES,
                   "max_rounds": MAX_ROUNDS, "round_ladder": ROUND_LADDER,
                   "refit_round_scale": REFIT_ROUND_SCALE},
    }

    # =====================================================================
    # 1. Per-regime hyper-parameter selection on inner holdouts
    # =====================================================================
    log("--- inner tuning: building the four regime-matched holdouts ---")
    fit_mask, devs, split_info = S3.inner_regime_splits(meta, masks)
    report["inner_split"] = split_info

    log("fitting inner encoders on inner_fit rows only ...")
    enc_in = F.EncoderSet(meta, Y, D, C, fit_mask, n_folds=F.N_FOLDS, seed=SEED)

    log("refitting both residual baselines on inner_fit rows only ...")
    mu_ctx_in, _, _ = VS.frozen_delta_baseline(meta, D, fit_mask, VS.CTX_LEVELS,
                                               "mu_ctx_inner")
    mu_drug_in, _, _ = VS.frozen_delta_baseline(meta, D, fit_mask, VS.DRUG_LEVELS,
                                                "mu_drug_inner")

    log("building the inner training matrix (unmasked; masks are applied per regime) ...")
    fit_idx = np.flatnonzero(fit_mask)

    def tune_mask_fn(blk: np.ndarray) -> np.ndarray:
        r = np.random.default_rng(SEED + 1000 + int(blk[0]))
        fin = np.isfinite(D[blk])
        return fin & (r.random(fin.shape) < INNER_TUNE_FRAC)

    Xin, _, yd_in = F.assemble_training_matrix(enc_in, fit_idx, tune_mask_fn,
                                              chunk=400, label="inner-fit")
    log(f"inner training matrix: {Xin.shape[0]:,} rows x {Xin.shape[1]} features")

    rng = np.random.default_rng(SEED)
    p = enc_in.p
    dev_data: dict[str, dict] = {}
    for regime in F.REGIMES:
        idx = np.flatnonzero(devs[regime])
        if len(idx) > N_DEV_SAMPLES:
            idx = np.sort(rng.choice(idx, size=N_DEV_SAMPLES, replace=False))
        log(f"building the '{regime}' dev design matrix ({len(idx)} samples) ...")
        Xd = build_dev_design(enc_in, idx, chunk=200, label=f"dev-{regime}")
        if len(Xd) != len(idx) * p:
            raise ValueError(f"dev design for {regime} has {len(Xd)} rows, "
                             f"expected {len(idx) * p}")
        dev_data[regime] = {"idx": idx, "X": Xd, "D": D[idx],
                            "mu_ctx": mu_ctx_in[idx], "mu_drug": mu_drug_in[idx]}

    trials: list[dict] = []
    selected: dict[str, dict] = {}
    for regime in F.REGIMES:
        log(f"=== tuning regime '{regime}' (masking: {F.REGIMES[regime] or 'nothing'}) ===")
        Xin_r = F.apply_regime_mask(Xin, regime, copy=True)
        dd = dev_data[regime]
        n_dev = len(dd["idx"])
        best_row = None
        for cfg in candidate_configs():
            t0 = time.time()
            model = fit_lgb(Xin_r, yd_in, cfg, MAX_ROUNDS)
            fit_s = time.time() - t0
            for k in ROUND_LADDER:
                Dhat = model.predict(dd["X"], num_iteration=k).astype("float32") \
                    .reshape(n_dev, p)
                met = regime_objective(regime, dd["D"], Dhat, dd["mu_ctx"], dd["mu_drug"])
                row = {"regime": regime, "candidate": cfg["name"], "n_iterations": k,
                       "n_dev_samples": n_dev, "fit_seconds": round(fit_s, 1),
                       "sel_objective": round(met["objective"], 6),
                       **{mk: round(mv, 6) for mk, mv in met.items() if mk != "objective"},
                       **{f"param_{pk}": pv for pk, pv in cfg.items() if pk != "name"}}
                trials.append(row)
                if best_row is None or row["sel_objective"] > best_row["sel_objective"]:
                    best_row = row
            log(f"  [tune:{regime}] '{cfg['name']}' best so far: "
                f"'{best_row['candidate']}' @{best_row['n_iterations']} "
                f"OBJ={best_row['sel_objective']:.4f} ({fit_s:.0f}s fit)")
        del Xin_r
        n_rounds = max(50, int(round(best_row["n_iterations"] * REFIT_ROUND_SCALE)))
        selected[regime] = {"candidate": best_row["candidate"],
                            "n_iterations_dev": best_row["n_iterations"],
                            "n_rounds_refit": n_rounds,
                            "dev_objective": best_row["sel_objective"],
                            "n_dev_samples": n_dev,
                            "config": {k: v for k, v in
                                       next(c for c in candidate_configs()
                                            if c["name"] == best_row["candidate"]).items()
                                       if k != "name"}}
        log(f"  [tune:{regime}] SELECTED '{best_row['candidate']}' @ "
            f"{best_row['n_iterations']} dev iters -> {n_rounds} refit rounds "
            f"(objective {best_row['sel_objective']:.4f})")

    tdf = pd.DataFrame(trials)
    tdf.to_csv(RESULTS / "gbdt_tuning_trials.csv", index=False)
    log(f"wrote results/gbdt_tuning_trials.csv ({len(tdf)} regime x candidate x iteration points)")
    for regime in F.REGIMES:
        sub = tdf[tdf["regime"] == regime].nlargest(3, "sel_objective")
        print(f"\n  top candidates for '{regime}':", flush=True)
        print(sub[["candidate", "n_iterations", "m2_pcc_per_sample", "sel_objective"]]
              .to_string(index=False), flush=True)

    report["tuning"] = {
        "design": ("one specialist per feature-availability regime. The OOD splits delete whole "
                   "feature families (measured in results/gbdt_feature_availability.json), so a "
                   "single model fitted on fully-featured train rows is served off-distribution "
                   "on the novel-strain splits. Each regime is tuned on its own holdout carved "
                   "from train and trained with exactly the features that regime can see."),
        "objective_definition": {
            "chem_novel": "(0.25 * m2_pcc + 0.20 * resid_pcc_vs_mu_ctx) / 0.45   [mirrors m3_s1_chem]",
            "strain_novel": "(0.25 * m2_pcc + 0.20 * resid_pcc_vs_mu_drug) / 0.45  [mirrors m3_s2_strain]",
            "both_novel": "(0.25 * m2_pcc + 0.05 * mean(m2, resid_ctx, resid_drug)) / 0.30  [m3_s3_both]",
            "full": "(0.25 * m2_pcc + 0.05 * mean(m2, resid_ctx, resid_drug)) / 0.30  [m3_time]",
        },
        "iteration_selection": ("the objective is evaluated at each point of ROUND_LADDER and "
                                "(config, n_iterations) is chosen jointly per regime, because the "
                                "rubric is a correlation and the L2-optimal iteration is not the "
                                "PCC-optimal one"),
        "round_ladder": ROUND_LADDER,
        "n_candidates": len(candidate_configs()),
        "n_trial_points": len(trials),
        "selected_per_regime": selected,
        "trials": trials,
        "note": ("candidates were ranked under the 'delta' formulation, which carries 80% of "
                 "the rubric weight; the winning configuration is reused for 'abs' rather "
                 "than tuned separately, to keep the two formulations comparable"),
    }
    del Xin, dev_data, enc_in, mu_ctx_in, mu_drug_in

    # =====================================================================
    # 2. Final fits: one specialist per (library, target, regime)
    # =====================================================================
    log("--- final training on all train rows ---")
    dm = pd.read_parquet(DATA / "gbdt_design_train.parquet")
    y_abs = dm.pop("y_abs").to_numpy("float32")
    y_delta = dm.pop("y_delta").to_numpy("float32")
    X = dm[F.FEATURE_NAMES]
    log(f"design matrix: {X.shape[0]:,} rows x {X.shape[1]} features "
        f"({X.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
    del dm

    targets = {"delta": y_delta, "abs": y_abs}
    fits: dict[str, object] = {}
    imp_rows: list[dict] = []

    for regime in F.REGIMES:
        n_rounds = selected[regime]["n_rounds_refit"]
        cfg = next(c for c in candidate_configs()
                   if c["name"] == selected[regime]["candidate"])
        log(f"=== regime '{regime}': config '{cfg['name']}', {n_rounds} rounds, "
            f"masking {F.REGIMES[regime] or 'nothing'} ===")
        Xr = F.apply_regime_mask(X, regime, copy=True)
        for target_kind, y in targets.items():
            for lib in ("lgb", "xgb", "cat"):
                name = f"{lib}_{target_kind}__{regime}"
                t0 = time.time()
                log(f"  [fit] {name}: {len(y):,} rows, {n_rounds} rounds ...")
                model = FITTERS[lib](Xr, y, cfg, n_rounds)
                secs = time.time() - t0

                path = MODELS / (f"{name}.txt" if lib == "lgb"
                                 else f"{name}.json" if lib == "xgb" else f"{name}.cbm")
                model.save_model(str(path))

                imp = importance(model, lib)
                for f, v in imp.items():
                    imp_rows.append({"model": f"{lib}_{target_kind}", "library": lib,
                                     "target": target_kind, "regime": regime,
                                     "feature": f, "gain_share": float(v)})
                top = imp.sort_values(ascending=False).head(6)
                fits[name] = {"library": lib, "target": target_kind, "regime": regime,
                              "config": cfg["name"], "n_rounds": n_rounds,
                              "fit_seconds": round(secs, 1),
                              "model_file": str(path.relative_to(WORKFLOW.parent)),
                              "top_features": {k: round(float(v), 4) for k, v in top.items()}}
                log(f"  [fit] {name} done in {secs:.0f}s | top gain: "
                    + ", ".join(f"{k}={v:.3f}" for k, v in top.items()))
        del Xr

    pd.DataFrame(imp_rows).to_csv(RESULTS / "gbdt_feature_importance.csv", index=False)
    log(f"wrote results/gbdt_feature_importance.csv ({len(imp_rows)} rows)")

    report["fits"] = fits
    report["n_models"] = len(fits)
    report["design_matrix"] = {"n_rows": int(X.shape[0]), "n_features": int(X.shape[1])}
    (RESULTS / "gbdt_training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    log("wrote results/gbdt_training_report.json")
    log(f"=== Step 3b complete in {time.time() - T0:.1f}s ===")


if __name__ == "__main__":
    main()
