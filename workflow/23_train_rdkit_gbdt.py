"""Step 4.2 -- retrain the GBDT regime specialists with RDKit molecular features.

This is a **controlled A/B experiment**, not a new tuning round. The rows, the
target, the hyper-parameters and the boosting-round count are taken verbatim
from the Step-3 selection (``results/gbdt_training_report.json``); the only thing
that changes is the feature set, which gains the RDKit descriptor +
fingerprint-PCA block. Any change in the S1 (``val_chem_only``) score is
therefore attributable to the molecular features rather than to re-tuning.

Why this can help at all
------------------------
Step 3 established that a novel compound carries *no* transferable information
through the categorical ``perturbation_no_concentration`` level (it maps to
``__UNSEEN__``) nor through the chemical-keyed group-mean features (blanked by
the ``chem_novel`` regime mask). The molecular block is attached *after* that
mask, so it is the only chemical signal a ``chem_novel`` specialist can see, and
it is defined for unseen compounds by construction.

Why it may nevertheless fail to help
------------------------------------
The effective sample size on the chemical axis is the number of *distinct
compounds* in training (37 in the inner fit, 46 overall), not the 5.6 M design
rows. A structure-activity relationship learned from 37 compounds is weakly
identified, so the honest prior is a small effect. This is recorded up front so
that a null result is reported as a finding rather than reframed after the fact.

Usage
-----
    uv run python workflow/23_train_rdkit_gbdt.py --libs lgb
    uv run python workflow/23_train_rdkit_gbdt.py --libs lgb,xgb --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step4_common as S4  # noqa: E402

SESSION = S4.SESSION
DATA, RESULTS, WORKFLOW = S4.DATA, S4.RESULTS, S4.WORKFLOW
MODELS4 = S4.MODELS4
SEED = S4.SEED
log = S4.log

REGIMES = ("full", "chem_novel", "strain_novel", "both_novel")


def selected_configs() -> dict[str, dict]:
    """Per-regime config and refit round count chosen by Step 3."""
    rep = json.loads((RESULTS / "gbdt_training_report.json").read_text(encoding="utf-8"))
    sel = rep["tuning"]["selected_per_regime"]
    return {
        r: {
            "config": sel[r]["config"],
            "n_rounds": int(sel[r]["n_rounds_refit"]),
            "candidate": sel[r]["candidate"],
        }
        for r in REGIMES
    }


def fit_lgb(X, y, cfg, n_rounds, cat_features, seed=SEED):
    """Fit a LightGBM booster at the Step-3 parameters (categoricals native)."""
    import lightgbm as lgb

    params = {k: v for k, v in cfg.items() if k != "name"}
    params.update(
        {
            "verbosity": -1,
            "num_threads": 24,
            "seed": seed,
            "deterministic": True,
            "force_row_wise": True,
            "cat_smooth": 20,
            "min_data_per_group": 100,
            "max_cat_threshold": 64,
        }
    )
    dtrain = lgb.Dataset(X, label=y, categorical_feature=cat_features, free_raw_data=False)
    return lgb.train(
        params, dtrain, num_boost_round=n_rounds, callbacks=[lgb.log_evaluation(period=200)]
    )


def fit_xgb(X, y, cfg, n_rounds, cat_features, seed=SEED):
    """Fit an XGBoost booster at the Step-3 parameters (native categoricals)."""
    import xgboost as xgb

    obj = "reg:pseudohubererror" if cfg.get("objective") == "huber" else "reg:squarederror"
    params = {
        "objective": obj,
        "tree_method": "hist",
        "max_bin": 256,
        "max_leaves": int(cfg["num_leaves"]),
        "grow_policy": "lossguide",
        "max_depth": 0,
        "eta": float(cfg["learning_rate"]),
        "min_child_weight": float(cfg["min_data_in_leaf"]) / 10.0,
        "colsample_bytree": float(cfg["feature_fraction"]),
        "subsample": float(cfg["bagging_fraction"]),
        "reg_lambda": float(cfg["lambda_l2"]),
        "max_cat_to_onehot": 8,
        "nthread": 24,
        "seed": seed,
    }
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True, nthread=24)
    return xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=200)


FITTERS = {"lgb": fit_lgb, "xgb": fit_xgb}
EXT = {"lgb": "txt", "xgb": "json"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", default="lgb", help="comma-separated: lgb,xgb")
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (row subsample + few rounds) to validate the code path",
    )
    args = ap.parse_args()
    libs = [s for s in args.libs.split(",") if s]
    regimes = [s for s in args.regimes.split(",") if s]

    np.random.seed(SEED)
    MODELS4.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(WORKFLOW))
    import features as F

    log("=== Step 4.2: RDKit-enhanced GBDT retraining ===")
    log(f"libs={libs} regimes={regimes} smoke={args.smoke}")

    mol = S4.load_mol_features("gbdt")
    log(f"molecular block: {mol.shape[1]} features x {mol.shape[0]} compounds")
    log(f"  columns: {list(mol.columns)}")

    log("loading the Step-3 design matrix (this is the same matrix Step 3 trained on) ...")
    dm = pd.read_parquet(DATA / "gbdt_design_train.parquet")
    y_delta = dm.pop("y_delta").to_numpy("float32")
    dm.pop("y_abs")
    # The compound name per row, read BEFORE any regime mask blanks it.
    chem_rows = dm[S4.CHEM_COL].astype(str).to_numpy()
    X = dm[F.FEATURE_NAMES]
    log(f"  design matrix {X.shape}, target y_delta finite={np.isfinite(y_delta).all()}")
    log(f"  distinct compounds in the training rows: {len(set(chem_rows.tolist()))}")

    if args.smoke:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(X), size=min(200_000, len(X)), replace=False)
        X, y_delta, chem_rows = X.iloc[keep], y_delta[keep], chem_rows[keep]
        log(f"  SMOKE: subsampled to {X.shape}")

    sel = selected_configs()
    report: dict = {
        "step": "4b_rdkit_gbdt",
        "seed": SEED,
        "design": (
            "controlled A/B against the Step-3 tabular specialists: identical rows, target, "
            "hyper-parameters and round count; the feature set alone differs"
        ),
        "smoke": bool(args.smoke),
        "n_rows": int(X.shape[0]),
        "n_tabular_features": int(X.shape[1]),
        "n_mol_features": int(mol.shape[1]),
        "mol_features": list(mol.columns),
        "n_distinct_train_compounds": len(set(chem_rows.tolist())),
        "effective_sample_size_caveat": (
            "the chemical axis has only ~46 independent observations (distinct compounds); "
            "a structure-activity relationship fitted from that many compounds is weakly "
            "identified, so a small or null effect on S1 is the expected outcome"
        ),
        "fits": [],
    }

    for regime in regimes:
        cfg, n_rounds = sel[regime]["config"], sel[regime]["n_rounds"]
        if args.smoke:
            n_rounds = 30
        log(f"--- regime '{regime}' | config '{sel[regime]['candidate']}' | rounds {n_rounds} ---")

        t0 = time.time()
        # Mask the tabular features this regime cannot see, THEN attach the
        # molecular block, which is available for novel compounds by design.
        Xr = F.apply_regime_mask(X, regime, copy=True)
        Xr = S4.attach_mol(Xr, chem_rows, mol)
        log(f"  design for '{regime}': {Xr.shape} ({time.time() - t0:.0f}s to build)")

        for lib in libs:
            name = f"{lib}_rdkit_delta__{regime}"
            path = MODELS4 / f"{name}.{EXT[lib]}"
            if path.exists() and not args.smoke:
                log(f"  {name} already trained; skipping")
                continue
            t1 = time.time()
            log(f"  fitting {name} ...")
            model = FITTERS[lib](Xr, y_delta, cfg, n_rounds, F.CAT_FEATURES)
            model.save_model(str(path))
            secs = time.time() - t1
            log(f"  saved {path.name} ({secs:.0f}s)")
            report["fits"].append(
                {
                    "name": name,
                    "lib": lib,
                    "regime": regime,
                    "candidate": sel[regime]["candidate"],
                    "n_rounds": n_rounds,
                    "n_features": int(Xr.shape[1]),
                    "seconds": round(secs, 1),
                    "path": str(path),
                }
            )
            del model

        del Xr

    out = RESULTS / "step4_rdkit_gbdt_training.json"
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
        report["fits"] = prev.get("fits", []) + report["fits"]
    S4.write_json(out, report)
    log(f"=== done: {len(report['fits'])} fits recorded ===")


if __name__ == "__main__":
    main()
