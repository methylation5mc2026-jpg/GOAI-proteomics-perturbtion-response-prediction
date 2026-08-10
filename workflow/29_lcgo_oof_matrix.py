"""Step 5.3 -- 5-fold Leave-Chemical-Group-Out out-of-fold prediction matrix.

Why replace Step 4's single inner holdout
-----------------------------------------
Step 4 fitted the stacking weights on one inner-dev cohort of 2,700 rows, using
members retrained on the 2,378 remaining train rows. Two defects follow directly
from that design, and this script exists to remove both:

1. **Weight-estimation variance.** 2,700 rows carried every one of the 16
   weights. The 5-fold design below gives an out-of-fold prediction for *all
   5,078* train rows, so the meta-learner sees 1.9x more calibration data -- and
   that matters much more once the weight tensor is expanded to per-cluster
   weights in Step 5.4, where there are ~10x more parameters to pin down.
2. **Member-strength mismatch.** Weights calibrated against a member fitted on
   2,378 rows are then applied to a member fitted on all 5,078. The 5-fold fit
   sets hold ~3,300 rows, roughly halving that gap. It does not close it -- no
   cross-fitting scheme can -- and the residual gap is reported rather than
   claimed away.

The fold construction
---------------------
A plain leave-chemical-group-out partition would make every held-out row
``chem_novel``, and the competition scores ``strain_novel`` at equal weight
(0.20 each). So chemicals are partitioned into 5 row-count-balanced groups **and**
each fold additionally holds out one strain, giving::

    fold f fit set  =  { rows : chem_group != f  AND  strain not held out by f }
                       minus the rows individually assigned to fold f's dev set

``train`` contains only 4 strains against 5 folds, so the 5 fold slots are dealt
round-robin over the strains: one strain is held out by two folds and every fold
holds out exactly one. (A one-to-one map instead leaves one fold holding out no
strain, which measurably skews the regime mix -- see ``strain_folds`` below.)

Each row is assigned to exactly one dev fold, and its regime follows from which
of its entities that fold holds out:

===============  ===========================================  ==================
regime           condition                                    mirrors
===============  ===========================================  ==================
``chem_novel``   assigned to fold ``chem_group``, which does   val_chem_only
                 not hold out the row's strain
``strain_novel`` assigned to a fold holding out the row's      val_strain_only
                 strain but not its chemical
``both_novel``   ``chem_group`` also holds out the strain      val_both
``full``         assigned to a fold holding out neither        val_time
===============  ===========================================  ==================

``full`` rows are a random carve (default 6%) withheld individually: their
chemical and strain are both still present in the fit set through *other* rows,
which is exactly the in-domain availability regime ``val_time`` represents.

Every one of those four claims is *verified*, not assumed: for each dev row the
audit recomputes entity novelty against the fold's realised fit set and asserts
it matches the assigned regime label. It also asserts the dev folds partition all
5,078 rows exactly once and that no fit/dev overlap exists. A violation raises.

Per fold, everything the objective touches is refitted on that fold's fit rows
only -- encoders, ``mu_ctx``, ``mu_drug``, the group-mean benchmark and the
abundance fallback -- so the assembled OOF cohort is honest end to end.

Stages
------
``--stage oof``      train the fold models, write the OOF member matrices
``--stage valtest``  train the full-train counterpart of any *new* family and
                     cache its ``val`` and ``test`` member matrices, so every
                     role exists on all three cohorts at matched
                     hyper-parameters (only the fit set differs)

Outputs
-------
data/step5_cache/oof_<family>.npy      5,078 x 5,243 fold-honest members
data/step5_cache/oof_baselines.npz     per-row mu_ctx / mu_drug / bench / fallback
results/step5_lcgo_folds.json          fold design and realised composition
results/step5_leakage_audit.json       the verification described above
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

DATA, RESULTS, WORKFLOW, MODELS4 = S4.DATA, S4.RESULTS, S4.WORKFLOW, S4.MODELS4
SEED, CHEM_COL, log = S4.SEED, S4.CHEM_COL, S4.log

CACHE5 = DATA / "step5_cache"
MODELS5 = WORKFLOW / "models_step5"
REGIMES = ("full", "chem_novel", "strain_novel", "both_novel")
N_FOLDS = 5
SUBSAMPLE_FRAC = 0.30  # identical to Step 3 / Step 4, so members stay comparable
FULL_FRAC = 0.06       # rows carved as the both-entities-seen ('full') regime

#: Target regime mix, taken from the official test split proportions so the OOF
#: objective resembles the objective it is calibrating for.
#: test_chem_only 1640 / test_strain_only 1322 / test_both 1129 / test_time 135.
TARGET_CHEM_SHARE = 1640 / (1640 + 1322)

#: family -> (library, molecular block). Ordered by priority: whatever the time
#: budget cannot reach is logged as skipped, never silently dropped.
FAMILIES: dict[str, dict] = {
    "lgb_tab": {"lib": "lgb", "mol": None},
    "lgb_mol": {"lib": "lgb", "mol": "2d"},
    "lgb_mol3d": {"lib": "lgb", "mol": "3d"},
    "xgb_mol": {"lib": "xgb", "mol": "2d"},
    "cat_mol": {"lib": "cat", "mol": "2d"},
}


# ---------------------------------------------------------------------------
# Molecular blocks
# ---------------------------------------------------------------------------
def mol_block(kind: str | None) -> pd.DataFrame | None:
    """Return the per-compound feature table for a family's molecular view."""
    if kind is None:
        return None
    m2 = S4.load_mol_features("gbdt")
    if kind == "2d":
        return m2
    m3 = pd.read_parquet(DATA / "step5_mol3d_features.parquet")
    if m3.index.name != CHEM_COL:
        m3 = m3.set_index(CHEM_COL)
    out = m2.join(m3.astype("float32"), how="left")
    missing = out.isna().any(axis=1)
    if missing.any():
        raise KeyError(
            f"{int(missing.sum())} compounds lack a 3D record: "
            f"{sorted(out.index[missing])[:8]}"
        )
    return out.astype("float32")


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------
def build_folds(meta: pd.DataFrame, train_mask: np.ndarray, seed: int = SEED) -> dict:
    """Partition the train rows into 5 LCGO dev folds with all four regimes.

    Returns a dict with ``fold`` and ``regime`` arrays over all rows (-1 / '' off
    the train split), the entity-to-group maps and the realised composition.
    """
    rng = np.random.default_rng(seed)
    tr_idx = np.flatnonzero(train_mask)
    chem = meta[CHEM_COL].astype(str).to_numpy()
    strain = meta["Strains"].astype(str).to_numpy()

    # ---- chemical groups, balanced by row count --------------------------
    counts = pd.Series(chem[tr_idx]).value_counts()
    chems = list(counts.index)
    rng.shuffle(chems)
    chems = sorted(chems, key=lambda c: -int(counts[c]))  # largest first
    chem_group: dict[str, int] = {}
    load = np.zeros(N_FOLDS, dtype=np.int64)
    for c in chems:
        g = int(np.argmin(load))
        chem_group[c] = g
        load[g] += int(counts[c])
    log(f"  {len(chems)} train chemicals -> {N_FOLDS} groups, row loads {load.tolist()}")

    # ---- strain fold-slots -----------------------------------------------
    # ``train`` holds only 4 strains but the design has 5 folds, so a one-to-one
    # strain->fold map leaves one fold holding out no strain at all. That fold
    # then contributes zero ``strain_novel`` and zero ``both_novel`` rows, and the
    # OOF regime mix drifts away from the official test mix it is meant to
    # calibrate against (measured: 18% both_novel against 27% in test).
    # Instead the 5 fold slots are dealt round-robin over the strains, so one
    # strain is held out in two folds and every fold holds out exactly one. A
    # strain may therefore appear in more than one fold's held-out set; that is
    # sound because each *row* is still assigned to exactly one dev fold.
    strains = sorted(set(strain[tr_idx]))
    perm = rng.permutation(len(strains))
    strain_folds: dict[str, list[int]] = {s: [] for s in strains}
    for f in range(N_FOLDS):
        strain_folds[strains[int(perm[f % len(strains)])]].append(f)
    log(f"  {len(strains)} train strains -> held-out folds {strain_folds}")

    g_of = np.full(len(meta), -1, dtype=np.int64)
    for i in tr_idx:
        g_of[i] = chem_group[chem[i]]
    sf_of: dict[int, set[int]] = {i: set(strain_folds[strain[i]]) for i in tr_idx}

    fold = np.full(len(meta), -1, dtype=np.int64)
    regime = np.full(len(meta), "", dtype=object)

    # ---- 'full' carve: a fold that holds out neither of the row's entities
    n_full = int(round(FULL_FRAC * len(tr_idx)))
    full_cand = np.array(
        [i for i in tr_idx if set(range(N_FOLDS)) - ({g_of[i]} | sf_of[i])]
    )
    full_rows = rng.choice(full_cand, size=min(n_full, len(full_cand)), replace=False)
    full_set = set(full_rows.tolist())
    for i in full_rows:
        cand = sorted(set(range(N_FOLDS)) - ({int(g_of[i])} | sf_of[i]))
        fold[i] = int(rng.choice(cand))
        regime[i] = "full"

    # ---- both_novel: the row's chemical fold also holds out its strain ----
    rest = np.array([i for i in tr_idx if i not in full_set])
    coincide = np.array([int(g_of[i]) in sf_of[i] for i in rest])
    for i in rest[coincide]:
        fold[i] = int(g_of[i])
        regime[i] = "both_novel"

    # ---- the remainder is split between chem_novel and strain_novel ------
    # Proportions follow the official test split so the OOF objective has a
    # similar module mix to the objective it calibrates for.
    split_rows = rest[~coincide]
    order = rng.permutation(len(split_rows))
    n_chem = int(round(TARGET_CHEM_SHARE * len(split_rows)))
    for j, pos in enumerate(order):
        i = split_rows[pos]
        if j < n_chem:
            fold[i] = int(g_of[i])
            regime[i] = "chem_novel"
        else:
            fold[i] = int(rng.choice(sorted(sf_of[i])))
            regime[i] = "strain_novel"

    assert (fold[tr_idx] >= 0).all(), "a train row was left unassigned"
    assert (regime[tr_idx] != "").all(), "a train row was left unlabelled"

    # ---- realised fit masks ---------------------------------------------
    held_strain_of_fold = {
        f: {s for s, fs in strain_folds.items() if f in fs} for f in range(N_FOLDS)
    }
    fits: list[np.ndarray] = []
    for f in range(N_FOLDS):
        is_hs = np.isin(strain, sorted(held_strain_of_fold[f]))
        m = train_mask & (g_of != f) & ~is_hs & (fold != f)
        fits.append(m)

    comp = {
        f"fold{f}": {
            "n_fit": int(fits[f].sum()),
            "n_dev": int((fold == f).sum()),
            "dev_by_regime": {
                r: int(((fold == f) & (regime == r)).sum()) for r in REGIMES
            },
            "n_fit_chemicals": int(len(set(chem[fits[f]]))),
            "n_fit_strains": int(len(set(strain[fits[f]]))),
            "held_out_chemicals": sorted({c for c, g in chem_group.items() if g == f}),
            "held_out_strains": sorted(held_strain_of_fold[f]),
        }
        for f in range(N_FOLDS)
    }
    for f in range(N_FOLDS):
        c = comp[f"fold{f}"]
        log(f"  fold {f}: fit n={c['n_fit']:5d} ({c['n_fit_chemicals']} chems, "
            f"{c['n_fit_strains']} strains)  dev n={c['n_dev']:5d} {c['dev_by_regime']}")

    overall = {r: int((regime == r).sum()) for r in REGIMES}
    log(f"  OOF regime mix: {overall}  (total {sum(overall.values())} of "
        f"{len(tr_idx)} train rows)")

    return {
        "fold": fold,
        "regime": regime,
        "fits": fits,
        "chem_group": chem_group,
        "strain_folds": {s: sorted(v) for s, v in strain_folds.items()},
        "g_of": g_of,
        "composition": comp,
        "overall_regime_mix": overall,
        "n_train": int(len(tr_idx)),
        "full_frac": FULL_FRAC,
        "target_chem_share_of_single_novelty_rows": TARGET_CHEM_SHARE,
    }


def audit_folds(meta: pd.DataFrame, folds: dict, train_mask: np.ndarray) -> dict:
    """Verify the fold design: partition, no overlap, regime labels correct.

    Recomputes entity novelty from each fold's realised fit set rather than
    trusting the construction, because a construction bug and a leakage bug look
    identical from the outside.
    """
    chem = meta[CHEM_COL].astype(str).to_numpy()
    strain = meta["Strains"].astype(str).to_numpy()
    fold, regime, fits = folds["fold"], folds["regime"], folds["fits"]
    tr_idx = np.flatnonzero(train_mask)

    checks: list[dict] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(ok), "detail": detail})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    assigned = np.zeros(len(meta), dtype=np.int64)
    for f in range(N_FOLDS):
        assigned += (fold == f).astype(np.int64)
    chk(
        "dev folds partition every train row exactly once",
        bool((assigned[tr_idx] == 1).all() and (assigned[~train_mask] == 0).all()),
        f"{int(assigned[tr_idx].sum())} assignments over {len(tr_idx)} train rows; "
        f"{int(assigned[~train_mask].sum())} non-train rows assigned",
    )

    n_overlap = sum(int((fits[f] & (fold == f)).sum()) for f in range(N_FOLDS))
    chk("no fold has a row in both its fit and dev set", n_overlap == 0,
        f"{n_overlap} overlapping rows")

    n_val_in_fit = sum(int((fits[f] & ~train_mask).sum()) for f in range(N_FOLDS))
    chk("no non-train (val) row enters any fit set", n_val_in_fit == 0,
        f"{n_val_in_fit} rows")

    # Regime labels re-derived from the realised fit sets.
    mismatch: list[dict] = []
    per_fold: dict[str, dict] = {}
    for f in range(N_FOLDS):
        seen_c = set(chem[fits[f]].tolist())
        seen_s = set(strain[fits[f]].tolist())
        dev = np.flatnonzero(fold == f)
        cnt = {r: 0 for r in REGIMES}
        for i in dev:
            cs = chem[i] in seen_c
            ss = strain[i] in seen_s
            exp = (
                "full" if (cs and ss)
                else "strain_novel" if (cs and not ss)
                else "chem_novel" if (not cs and ss)
                else "both_novel"
            )
            cnt[regime[i]] += 1
            if exp != regime[i]:
                mismatch.append(
                    {"row": int(i), "fold": f, "assigned": regime[i], "derived": exp,
                     "chemical": chem[i], "strain": strain[i]}
                )
        per_fold[f"fold{f}"] = {
            "n_fit_chemicals": len(seen_c),
            "n_fit_strains": len(seen_s),
            "dev_regime_counts": cnt,
        }
    chk(
        "every dev row's regime label matches entity novelty against its fold's fit set",
        len(mismatch) == 0,
        f"{len(mismatch)} mismatched rows"
        + (f", e.g. {mismatch[:3]}" if mismatch else ""),
    )

    # Chemical leakage, stated the way the plan states it.
    leaks = []
    for f in range(N_FOLDS):
        seen_c = set(chem[fits[f]].tolist())
        dev = np.flatnonzero(fold == f)
        novel = [i for i in dev if regime[i] in ("chem_novel", "both_novel")]
        bad = [i for i in novel if chem[i] in seen_c]
        if bad:
            leaks.append({"fold": f, "n_leaked": len(bad)})
    chk("zero chemical leakage into any chem-novel dev row", len(leaks) == 0,
        f"{sum(l['n_leaked'] for l in leaks)} leaked rows")

    n_pass = sum(c["passed"] for c in checks)
    out = {
        "step": "5_3_leakage_audit",
        "seed": SEED,
        "n_checks": len(checks),
        "n_passed": n_pass,
        "all_passed": n_pass == len(checks),
        "checks": checks,
        "per_fold": per_fold,
        "n_regime_mismatches": len(mismatch),
        "regime_mismatch_examples": mismatch[:20],
        "note": (
            "regime labels are re-derived from each fold's realised fit set rather than "
            "trusted from the construction, so a construction bug cannot hide as a pass"
        ),
    }
    if not out["all_passed"]:
        raise AssertionError(
            f"LCGO fold audit failed: {json.dumps([c for c in checks if not c['passed']], indent=2)}"
        )
    return out


# ---------------------------------------------------------------------------
# Per-fold refits
# ---------------------------------------------------------------------------
def refit_baselines(ctx: dict, fit_mask: np.ndarray, tag: str) -> dict:
    """Refit every scoring baseline an OOF row's score depends on, on fit rows."""
    VS = ctx["VS"]
    meta, Y, D = ctx["meta"], ctx["Y"], ctx["D"]
    mu_ctx, _, _ = VS.frozen_delta_baseline(meta, D, fit_mask, VS.CTX_LEVELS, f"{tag}_mu_ctx")
    mu_drug, _, _ = VS.frozen_delta_baseline(meta, D, fit_mask, VS.DRUG_LEVELS, f"{tag}_mu_drug")
    Y_bench, _, _ = VS.frozen_delta_baseline(
        meta, Y, fit_mask, VS.CTX_LEVELS_BATCH, f"{tag}_bench"
    )
    y_fb = S4._abundance_fallback(meta, Y, fit_mask, VS)
    return {"mu_ctx": mu_ctx, "mu_drug": mu_drug, "Y_bench": Y_bench, "y_fallback": y_fb}


def selected_configs() -> dict[str, tuple[dict, int]]:
    """Per-regime LightGBM config and round count chosen by Step 3 (verbatim).

    Reused unchanged so that a Step-5 member differs from its Step-4 counterpart
    only in the fit set. Re-tuning here would confound the cross-fitting change
    with a hyper-parameter change.
    """
    rep = json.loads((RESULTS / "gbdt_training_report.json").read_text(encoding="utf-8"))
    sel = rep["tuning"]["selected_per_regime"]
    return {r: (sel[r]["config"], int(sel[r]["n_rounds_refit"])) for r in REGIMES}


def cat_frame(X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    """CatBoost view: categoricals as integer codes (avoids millions of strings)."""
    out = X.copy()
    for c in cat_features:
        out[c] = out[c].cat.codes.astype("int32")
    return out


def fit_family(X: pd.DataFrame, y: np.ndarray, lib: str, cfg: dict, n_rounds: int,
               cat_features: list[str]):
    """Dispatch to the library-specific fitter at the Step-3 hyper-parameters."""
    if lib == "lgb":
        import lightgbm as lgb

        params = {k: v for k, v in cfg.items() if k != "name"}
        params.update({"verbosity": -1, "num_threads": 24, "seed": SEED,
                       "deterministic": True, "force_row_wise": True,
                       "cat_smooth": 20, "min_data_per_group": 100,
                       "max_cat_threshold": 64})
        ds = lgb.Dataset(X, label=y, categorical_feature=cat_features, free_raw_data=False)
        return lgb.train(params, ds, num_boost_round=n_rounds)
    if lib == "xgb":
        import xgboost as xgb

        params = {
            "objective": "reg:squarederror", "tree_method": "hist", "max_bin": 256,
            "max_leaves": int(cfg["num_leaves"]), "grow_policy": "lossguide",
            "max_depth": 0, "eta": float(cfg["learning_rate"]),
            "min_child_weight": float(cfg["min_data_in_leaf"]) / 10.0,
            "colsample_bytree": float(cfg["feature_fraction"]),
            "subsample": float(cfg["bagging_fraction"]),
            "reg_lambda": float(cfg["lambda_l2"]),
            "max_cat_to_onehot": 8, "nthread": 24, "seed": SEED,
        }
        d = xgb.DMatrix(X, label=y, enable_categorical=True, nthread=24)
        return xgb.train(params, d, num_boost_round=n_rounds, verbose_eval=False)
    from catboost import CatBoostRegressor, Pool

    pool = Pool(cat_frame(X, cat_features), label=y, cat_features=cat_features)
    m = CatBoostRegressor(
        iterations=n_rounds, learning_rate=float(cfg["learning_rate"]),
        depth=min(10, max(4, int(np.log2(cfg["num_leaves"])) + 1)),
        l2_leaf_reg=float(cfg["lambda_l2"]), loss_function="RMSE",
        rsm=float(cfg["feature_fraction"]), random_seed=SEED,
        min_data_in_leaf=int(cfg["min_data_in_leaf"]), border_count=128,
        thread_count=24, verbose=False, allow_writing_files=False,
        bootstrap_type="Bernoulli", subsample=float(cfg["bagging_fraction"]),
    )
    m.fit(pool)
    return m


def design_for(ctx: dict, enc, fit_idx: np.ndarray, chunk: int = 300,
               label: str = "design") -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Assemble a fold's long design matrix with the Step-3 cell subsampling.

    Also returns the per-row compound name, recovered by replaying the identical
    seeded cell mask; the encoder maps novel chemicals to ``__UNSEEN__`` so the
    name cannot be read back off the design matrix.
    """
    F, D = ctx["F"], ctx["D"]
    meta = ctx["meta"]

    def cell_mask_fn(block_idx: np.ndarray) -> np.ndarray:
        fin = np.isfinite(D[block_idx])
        r = np.random.default_rng(SEED + int(block_idx[0]))
        return fin & (r.random(fin.shape) < SUBSAMPLE_FRAC)

    t0 = time.time()
    X, _, y_delta = F.assemble_training_matrix(
        enc, fit_idx, cell_mask_fn, chunk=chunk, label=label
    )
    counts = []
    for s in range(0, len(fit_idx), chunk):
        blk = fit_idx[s : s + chunk]
        counts.append(cell_mask_fn(blk).sum(axis=1))
    counts = np.concatenate(counts)
    if int(counts.sum()) != len(X):
        raise AssertionError(
            f"cell-mask replay mismatch: {int(counts.sum())} vs {len(X)} design rows; "
            "the per-row compound mapping would be misaligned"
        )
    chem_rows = np.repeat(meta.iloc[fit_idx][CHEM_COL].astype(str).to_numpy(), counts)
    log(f"  design {X.shape} in {time.time() - t0:.0f}s, "
        f"{len(set(chem_rows.tolist()))} compounds")
    assert np.isfinite(y_delta).all(), "non-finite Delta target survived the cell mask"
    return X, y_delta, chem_rows


# ---------------------------------------------------------------------------
# Stage: OOF
# ---------------------------------------------------------------------------
def stage_oof(args) -> None:
    """Train the fold models and assemble the full-length OOF member matrices."""
    sys.path.insert(0, str(WORKFLOW))
    import features as F

    CACHE5.mkdir(parents=True, exist_ok=True)
    MODELS5.mkdir(parents=True, exist_ok=True)

    ctx = S4.load_context()
    meta, Y, D, C_h = ctx["meta"], ctx["Y"], ctx["D"], ctx["C_harness"]
    masks, VS = ctx["masks"], ctx["VS"]
    train_mask = masks[VS.TRAIN_SPLIT]
    n_all, n_p = Y.shape

    log("=== building the 5-fold LCGO design ===")
    folds = build_folds(meta, train_mask, seed=SEED)
    log("auditing the fold design ...")
    audit = audit_folds(meta, folds, train_mask)
    S4.write_json(RESULTS / "step5_leakage_audit.json", audit)

    fam_names = [f for f in args.families.split(",") if f]
    for f in fam_names:
        if f not in FAMILIES:
            raise SystemExit(f"unknown family {f!r}; known: {sorted(FAMILIES)}")
    log(f"families requested (in priority order): {fam_names}")

    if args.dry_run:
        # Verify the molecular blocks resolve and report the design, then stop --
        # a fold-design bug should surface in seconds, not after an hour of fits.
        for fam in fam_names:
            mb = mol_block(FAMILIES[fam]["mol"])
            log(f"  {fam}: molecular block "
                f"{'none (37 tabular features)' if mb is None else str(mb.shape)}")
        S4.write_json(
            RESULTS / "step5_lcgo_folds.json",
            {"step": "5_3_lcgo_folds", "dry_run": True, "seed": SEED,
             "n_folds": N_FOLDS, "composition": folds["composition"],
             "overall_regime_mix": folds["overall_regime_mix"],
             "chem_group": folds["chem_group"], "strain_folds": folds["strain_folds"]},
        )
        log("=== dry run complete (fold design audited, no model trained) ===")
        return

    tr_idx = np.flatnonzero(train_mask)
    pos_of_row = -np.ones(n_all, dtype=np.int64)
    pos_of_row[tr_idx] = np.arange(len(tr_idx))

    # OOF accumulators, indexed by position within the train split.
    oof = {f: np.full((len(tr_idx), n_p), np.nan, dtype="float32") for f in fam_names}
    oof_bench = np.full((len(tr_idx), n_p), np.nan, dtype="float32")
    base = {
        k: np.full((len(tr_idx), n_p), np.nan, dtype="float32")
        for k in ("mu_ctx", "mu_drug", "y_fallback")
    }
    sel = selected_configs()
    timing: dict = {}
    skipped: dict = {}

    for f in range(N_FOLDS):
        fit_mask = folds["fits"][f]
        dev_rows = np.flatnonzero(folds["fold"] == f)
        dev_pos = pos_of_row[dev_rows]
        log(f"\n===== FOLD {f}: fit n={int(fit_mask.sum())}, dev n={len(dev_rows)} =====")
        t_fold = time.time()

        log("refitting the residual baselines / benchmark / fallback on fit rows ...")
        bl = refit_baselines(ctx, fit_mask, f"fold{f}")
        for k in ("mu_ctx", "mu_drug", "y_fallback"):
            base[k][dev_pos] = bl[k][dev_rows]
        oof_bench[dev_pos] = np.nan_to_num(
            (bl["Y_bench"][dev_rows] - C_h[dev_rows]).astype("float32"),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        log("fitting fold encoders on fit rows only ...")
        enc = F.EncoderSet(meta, Y, D, ctx["C"], fit_mask, n_folds=F.N_FOLDS,
                           seed=SEED, verbose=False)

        fit_idx = np.flatnonzero(fit_mask)
        X, y_delta, chem_rows = design_for(ctx, enc, fit_idx, label=f"fold{f}_design")

        # ---- train, grouped by regime then by molecular view -------------
        models: dict[str, dict] = {}   # family -> {regime: (model, lib)}
        budget_hit = False
        for regime in REGIMES:
            if regime not in set(folds["regime"][dev_rows]):
                log(f"  regime '{regime}' absent from fold {f}'s dev set; specialist skipped")
                continue
            cfg, n_rounds = sel[regime]
            Xr = F.apply_regime_mask(X, regime, copy=True)
            for fam in fam_names:
                if budget_hit:
                    skipped.setdefault(f"fold{f}", []).append(f"{fam}/{regime}")
                    continue
                spec = FAMILIES[fam]
                mb = mol_block(spec["mol"])
                Xf = Xr if mb is None else S4.attach_mol(Xr, chem_rows, mb)
                t0 = time.time()
                m = fit_family(Xf, y_delta, spec["lib"], cfg, n_rounds, F.CAT_FEATURES)
                dt = time.time() - t0
                timing[f"fold{f}/{fam}/{regime}"] = round(dt, 1)
                log(f"  fold {f} {fam:10s} {regime:13s} fit {dt:6.0f}s "
                    f"({Xf.shape[1]} features, {n_rounds} rounds)")
                models.setdefault(fam, {})[regime] = (m, spec["lib"])
                if mb is None:
                    del Xf
                if time.time() - t_fold > args.fold_budget:
                    budget_hit = True
                    log(f"  !! fold time budget {args.fold_budget:.0f}s reached; "
                        f"remaining families in this fold will be skipped and logged")
            del Xr
        del X

        # ---- predict the dev rows, one molecular view at a time -----------
        by_view: dict[str | None, list[str]] = {}
        for fam in models:
            by_view.setdefault(FAMILIES[fam]["mol"], []).append(fam)
        for view, fams in by_view.items():
            mb = mol_block(view)
            sub = {fam: models[fam] for fam in fams}
            # A family whose specialist is missing for some regime cannot be
            # predicted by predict_mol_families (it indexes by regime), so those
            # families are handled one regime at a time below instead.
            complete = {
                fam: d for fam, d in sub.items()
                if set(d) >= set(np.unique(folds["regime"][dev_rows]).tolist())
            }
            partial = {fam: d for fam, d in sub.items() if fam not in complete}
            if complete:
                raw, _ = S4.predict_mol_families(
                    complete, enc, dev_rows, meta, mb, chunk=250,
                    label=f"fold{f}_{view or 'tab'}",
                )
                for fam, arr in raw.items():
                    oof[fam][dev_pos] = arr
            for fam, d in partial.items():
                for regime, (m, lib) in d.items():
                    rows = dev_rows[folds["regime"][dev_rows] == regime]
                    if not len(rows):
                        continue
                    raw, _ = S4.predict_mol_families(
                        {fam: {regime: (m, lib)}}, enc, rows, meta, mb, chunk=250,
                        label=f"fold{f}_{fam}_{regime}",
                    )
                    oof[fam][pos_of_row[rows]] = raw[fam]

        for fam, d in models.items():
            for regime, (m, lib) in d.items():
                p = MODELS5 / f"fold{f}__{fam}__{regime}"
                if lib == "lgb":
                    m.save_model(str(p) + ".txt")
                elif lib == "xgb":
                    m.save_model(str(p) + ".json")
                else:
                    m.save_model(str(p) + ".cbm")
        del models, enc
        log(f"===== FOLD {f} done in {time.time() - t_fold:.0f}s =====")

    # ---- persist ---------------------------------------------------------
    log("\nwriting the OOF member matrices ...")
    coverage: dict = {}
    for fam, arr in oof.items():
        nfin = int(np.isfinite(arr).all(axis=1).sum())
        coverage[fam] = {
            "n_rows_with_complete_prediction": nfin,
            "n_rows_total": int(len(tr_idx)),
            "frac_complete": nfin / len(tr_idx),
        }
        np.save(CACHE5 / f"oof_{fam}.npy", np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        log(f"  oof_{fam}.npy {arr.shape}  complete rows {nfin}/{len(tr_idx)}")
    np.save(CACHE5 / "oof_bench.npy", np.nan_to_num(oof_bench, nan=0.0))
    np.savez_compressed(
        CACHE5 / "oof_baselines.npz",
        mu_ctx=base["mu_ctx"], mu_drug=base["mu_drug"], y_fallback=base["y_fallback"],
        train_idx=tr_idx, regime=np.array(folds["regime"][tr_idx].tolist(), dtype=str),
        fold=folds["fold"][tr_idx],
    )
    log(f"  wrote {CACHE5 / 'oof_baselines.npz'}")

    S4.write_json(
        RESULTS / "step5_lcgo_folds.json",
        {
            "step": "5_3_lcgo_folds",
            "seed": SEED,
            "n_folds": N_FOLDS,
            "subsample_frac": SUBSAMPLE_FRAC,
            "full_frac": FULL_FRAC,
            "n_train_rows": int(len(tr_idx)),
            "families_requested": fam_names,
            "families_trained": sorted(oof),
            "chem_group": folds["chem_group"],
            "strain_folds": folds["strain_folds"],
            "composition": folds["composition"],
            "overall_regime_mix": folds["overall_regime_mix"],
            "oof_coverage": coverage,
            "fit_seconds": timing,
            "skipped_due_to_fold_time_budget": skipped,
            "fold_time_budget_seconds": args.fold_budget,
            "hyperparameters": {
                r: {"config": sel[r][0], "n_rounds": sel[r][1]} for r in REGIMES
            },
            "protocol_notes": [
                "hyper-parameters are the Step-3 per-regime selections, reused verbatim, so a "
                "Step-5 member differs from its Step-4 counterpart only in the fit set",
                "encoders, mu_ctx, mu_drug, the group-mean benchmark and the abundance "
                "fallback are all refitted on each fold's fit rows only",
                "fit sets hold ~3,300 rows against 5,078 for the final members, so a residual "
                "member-strength mismatch remains; it is roughly half the Step-4 gap",
            ],
        },
    )
    log("=== stage oof complete ===")


# ---------------------------------------------------------------------------
# Stage: val / test counterparts of new families
# ---------------------------------------------------------------------------
def stage_valtest(args) -> None:
    """Train full-train specialists for new families; cache val and test members."""
    sys.path.insert(0, str(WORKFLOW))
    import features as F

    CACHE5.mkdir(parents=True, exist_ok=True)
    MODELS5.mkdir(parents=True, exist_ok=True)
    fam_names = [f for f in args.families.split(",") if f]

    ctx = S4.load_context()
    meta = ctx["meta"]
    masks, VS, S3 = ctx["masks"], ctx["VS"], ctx["S3"]
    enc, proteins = ctx["enc"], ctx["proteins"]
    train_mask = masks[VS.TRAIN_SPLIT]
    sel = selected_configs()

    need_train = [
        f for f in fam_names
        if not all((MODELS5 / f"full__{f}__{r}.txt").exists() for r in REGIMES)
    ]
    log(f"families needing a full-train fit: {need_train}")

    models: dict[str, dict] = {f: {} for f in fam_names}
    if need_train:
        fit_idx = np.flatnonzero(train_mask)
        X, y_delta, chem_rows = design_for(ctx, enc, fit_idx, label="fulltrain_design")
        for regime in REGIMES:
            cfg, n_rounds = sel[regime]
            Xr = F.apply_regime_mask(X, regime, copy=True)
            for fam in need_train:
                spec = FAMILIES[fam]
                mb = mol_block(spec["mol"])
                Xf = Xr if mb is None else S4.attach_mol(Xr, chem_rows, mb)
                t0 = time.time()
                m = fit_family(Xf, y_delta, spec["lib"], cfg, n_rounds, F.CAT_FEATURES)
                log(f"  full-train {fam:10s} {regime:13s} fit {time.time() - t0:6.0f}s")
                p = MODELS5 / f"full__{fam}__{regime}"
                if spec["lib"] == "lgb":
                    m.save_model(str(p) + ".txt")
                elif spec["lib"] == "xgb":
                    m.save_model(str(p) + ".json")
                else:
                    m.save_model(str(p) + ".cbm")
                models[fam][regime] = (m, spec["lib"])
            del Xr
        del X
    for fam in fam_names:
        if len(models[fam]) == len(REGIMES):
            continue
        import lightgbm as lgb

        for regime in REGIMES:
            p = MODELS5 / f"full__{fam}__{regime}.txt"
            if p.exists():
                models[fam][regime] = (lgb.Booster(model_file=str(p)), "lgb")
        log(f"  loaded cached full-train specialists for {fam}")

    # ---- val cohort ------------------------------------------------------
    val_mask = masks["all_val"]
    val_idx = np.flatnonzero(val_mask)
    log(f"predicting the val cohort ({len(val_idx)} samples) ...")
    by_view: dict[str | None, list[str]] = {}
    for fam in fam_names:
        by_view.setdefault(FAMILIES[fam]["mol"], []).append(fam)
    for view, fams in by_view.items():
        mb = mol_block(view)
        raw, _ = S4.predict_mol_families(
            {f: models[f] for f in fams}, enc, val_idx, meta, mb, chunk=250,
            label=f"val_{view or 'tab'}",
        )
        for fam, arr in raw.items():
            np.save(CACHE5 / f"val_{fam}.npy", np.nan_to_num(arr, nan=0.0))
            log(f"  cached val_{fam}.npy {arr.shape}")

    # ---- test cohort -----------------------------------------------------
    log("loading the test cohort ...")
    te = S3.load_test(proteins, ctx["meta_all"], ctx["M_all"])
    meta_te, C_te = te["meta"], te["C"]
    ext = {"meta": meta_te, "C": C_te, "Y": None, "D": None}
    te_idx = np.arange(len(meta_te))
    for view, fams in by_view.items():
        mb = mol_block(view)
        raw, te_regimes = S4.predict_mol_families(
            {f: models[f] for f in fams}, enc, te_idx, meta_te, mb, external=ext,
            chunk=250, label=f"test_{view or 'tab'}",
        )
        for fam, arr in raw.items():
            np.save(CACHE5 / f"test_{fam}.npy", np.nan_to_num(arr, nan=0.0))
            log(f"  cached test_{fam}.npy {arr.shape}")
    np.save(CACHE5 / "test_regimes.npy", te_regimes.astype(str))
    log("=== stage valtest complete ===")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["oof", "valtest"], default="oof")
    ap.add_argument(
        "--families", default="lgb_tab,lgb_mol,lgb_mol3d",
        help="comma-separated family names in priority order",
    )
    ap.add_argument(
        "--fold-budget", type=float, default=5400.0,
        help="wall-clock seconds per fold before remaining families are skipped and logged",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="build and audit the fold design, resolve the feature blocks, then stop",
    )
    args = ap.parse_args()
    np.random.seed(SEED)
    log(f"=== Step 5.3: LCGO OOF matrix (stage={args.stage}) ===")
    if args.stage == "oof":
        stage_oof(args)
    else:
        stage_valtest(args)


if __name__ == "__main__":
    main()
