"""Step 4.4b -- cache the validation-cohort member predictions.

Step 3 persisted only *metrics* for the validation cohort, never the prediction
matrices, so any blending or stacking work has to regenerate them. This script
produces and caches one fold-change (Delta) matrix per member on the 2,806
``all_val`` rows, so the stacking script can evaluate a candidate weight vector
in milliseconds instead of re-running a 25-minute prediction pass per trial.

Members cached
--------------
``val_lgb_delta``        Step-3 tabular LightGBM regime specialists
``val_xgb_delta``        Step-3 tabular XGBoost regime specialists
``val_cat_delta``        Step-3 tabular CatBoost regime specialists
``val_lgb_rdkit_delta``  RDKit-enhanced LightGBM specialists (Step 4.2)
``val_xgb_rdkit_delta``  RDKit-enhanced XGBoost specialists (Step 4.2)
``val_bench_delta``      group-mean benchmark, train-frozen
plus ``val_C_harness``, ``val_y_fallback`` and ``val_regimes`` so downstream
scripts need not rebuild them.

All members are produced by the same routing rule (entity novelty) and the same
encoder as Step 3, so the cached matrices are directly comparable to the Step-3
reported scores. The script re-scores ``val_lgb_delta`` against the Step-3 number
as a self-check that nothing has drifted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_common as S4  # noqa: E402

DATA, RESULTS, WORKFLOW, MODELS4 = S4.DATA, S4.RESULTS, S4.WORKFLOW, S4.MODELS4
SEED, log = S4.SEED, S4.log
REGIMES = ("full", "chem_novel", "strain_novel", "both_novel")


def load_step3(lib: str) -> dict | None:
    """Load the four Step-3 regime specialists for one library, if present."""
    import catboost
    import lightgbm as lgb
    import xgboost as xgb

    ext = {"lgb": "txt", "xgb": "json", "cat": "cbm"}[lib]
    got = {}
    for regime in REGIMES:
        p = WORKFLOW / "models" / f"{lib}_delta__{regime}.{ext}"
        if not p.exists():
            return None
        if lib == "lgb":
            m = lgb.Booster(model_file=str(p))
        elif lib == "xgb":
            m = xgb.Booster()
            m.load_model(str(p))
        else:
            m = catboost.CatBoostRegressor()
            m.load_model(str(p))
        got[regime] = (m, lib)
    return got


def load_step4(lib: str) -> dict | None:
    """Load the four RDKit-enhanced regime specialists for one library."""
    import lightgbm as lgb
    import xgboost as xgb

    ext = {"lgb": "txt", "xgb": "json"}[lib]
    got = {}
    for regime in REGIMES:
        p = MODELS4 / f"{lib}_rdkit_delta__{regime}.{ext}"
        if not p.exists():
            return None
        if lib == "lgb":
            m = lgb.Booster(model_file=str(p))
        else:
            m = xgb.Booster()
            m.load_model(str(p))
        got[regime] = (m, lib)
    return got


def main() -> None:
    np.random.seed(SEED)
    log("=== Step 4.4b: caching validation-cohort member predictions ===")

    ctx = S4.load_context()
    meta, Y, D = ctx["meta"], ctx["Y"], ctx["D"]
    masks, VS = ctx["masks"], ctx["VS"]
    enc = ctx["enc"]

    eval_mask = masks["all_val"]
    eval_idx = np.flatnonzero(eval_mask)
    C_h = ctx["C_harness"][eval_mask]
    y_fb = ctx["y_fallback"][eval_mask]
    log(f"validation cohort: {len(eval_idx)} samples x {enc.p} proteins")

    S4.cache_put("val_C_harness", C_h)
    S4.cache_put("val_y_fallback", y_fb)

    # ---- analytic benchmark ---------------------------------------------
    bench = np.ascontiguousarray(ctx["Y_bench"][eval_mask], dtype="float32")
    S4.cache_put("val_bench_delta", (bench - C_h).astype("float32"))
    log(f"  benchmark delta defined on {100 * np.isfinite(bench - C_h).mean():.2f}% of cells")

    # ---- tabular Step-3 members -----------------------------------------
    mol = S4.load_mol_features("gbdt")
    todo_tab = {}
    for lib in ("lgb", "xgb", "cat"):
        if S4.cache_path(f"val_{lib}_delta").exists():
            log(f"  val_{lib}_delta cached; skipping")
            continue
        fam = load_step3(lib)
        if fam is None:
            log(f"  Step-3 {lib} specialists absent; skipping")
            continue
        todo_tab[f"val_{lib}_delta"] = fam

    if todo_tab:
        log(f"predicting {len(todo_tab)} tabular families (37 features, no mol block) ...")
        raw, regimes = S4.predict_mol_families(
            todo_tab, enc, eval_idx, meta, None, chunk=250, label="val_tab"
        )
        for nm, arr in raw.items():
            S4.cache_put(nm, arr)
        np.save(S4.cache_path("val_regimes"), regimes.astype(str))

    # ---- RDKit Step-4 members -------------------------------------------
    todo_mol = {}
    for lib in ("lgb", "xgb"):
        if S4.cache_path(f"val_{lib}_rdkit_delta").exists():
            log(f"  val_{lib}_rdkit_delta cached; skipping")
            continue
        fam = load_step4(lib)
        if fam is None:
            log(f"  RDKit {lib} specialists absent; skipping")
            continue
        todo_mol[f"val_{lib}_rdkit_delta"] = fam

    if todo_mol:
        log(f"predicting {len(todo_mol)} RDKit families (82 features, mol block attached) ...")
        raw, regimes = S4.predict_mol_families(
            todo_mol, enc, eval_idx, meta, mol, chunk=250, label="val_rdkit"
        )
        for nm, arr in raw.items():
            S4.cache_put(nm, arr)
        np.save(S4.cache_path("val_regimes"), regimes.astype(str))

    # ---- self-check: reproduce the Step-3 lgb_delta total score ----------
    import json

    meta_eval = meta.loc[eval_mask].reset_index(drop=True)
    Y_ev, D_ev = Y[eval_mask], D[eval_mask]
    mu_ctx_ev, mu_drug_ev = ctx["mu_ctx"][eval_mask], ctx["mu_drug"][eval_mask]

    d_lgb = S4.cache_get("val_lgb_delta")
    if d_lgb is not None:
        yh, dh = S4.reconstruct(d_lgb, C_h, y_fb)
        res = S4.score(Y_ev, yh, D_ev, dh, meta_eval, mu_ctx_ev, mu_drug_ev)
        got = float(res["total_score"])
        ref = json.loads((RESULTS / "gbdt_model_scores.json").read_text(encoding="utf-8"))
        want = float(ref["model_totals"]["lgb_delta"]) if "model_totals" in ref else None
        log(f"  self-check lgb_delta total: regenerated {got:.6f} vs Step-3 {want}")
        if want is not None:
            assert abs(got - want) < 2e-6, (
                f"regenerated lgb_delta total {got:.8f} != Step-3 {want:.8f}; the cached "
                "members are not the matrices Step 3 scored, so any stacked score built on "
                "them would be incomparable"
            )
            log("  self-check PASSED (cached members reproduce the Step-3 score exactly)")

    S4.write_json(
        RESULTS / "step4_val_members.json",
        {
            "step": "4e_val_members",
            "seed": SEED,
            "n_val_samples": int(len(eval_idx)),
            "n_proteins": int(enc.p),
            "cached": sorted(p.stem for p in S4.CACHE.glob("val_*.npy")),
            "self_check_lgb_delta_total": got if d_lgb is not None else None,
        },
    )
    log("=== validation members cached ===")


if __name__ == "__main__":
    main()
