"""Step 4.4a -- honest out-of-fold member predictions on the inner-dev cohort.

Why this script has to exist
----------------------------
The stacking weights must be fitted on predictions from models that have **not
seen the rows being predicted**. The Step-3 specialists in ``workflow/models/``
were trained on *all* train rows, and the inner-dev cohort is a subset of train,
so using them here would feed the meta-learner in-sample predictions. A member
that is in-sample looks far more accurate than it really is and would be handed
an inflated weight, which is exactly the failure mode stacking is supposed to
avoid.

So the GBDT members are retrained from scratch on ``inner_fit`` rows only, with
encoders and baselines also refitted on ``inner_fit`` (done by
:func:`step4_common.load_inner_context`). Their predictions on ``inner_dev`` are
then genuinely out-of-fold and the weights derived from them are honest.

Members produced here (all as fold-change / Delta matrices on inner-dev):
  * ``inner_lgb``        tabular LightGBM regime specialists
  * ``inner_lgb_rdkit``  the same, plus the RDKit molecular block
  * ``inner_bench``      the group-mean benchmark, refitted on inner_fit
  * (``inner_dl``        produced by ``24_train_deep_learning.py``)

The control anchor (Delta = 0) is not stored: it is the origin of the weight
space, so a non-negative combination whose weights sum to less than one already
represents shrinkage toward it.

A caveat that is reported rather than hidden: ``inner_fit`` has only ~2,378 rows
against ~5,078 for the full train split, so the inner members are weaker than the
final ones. Weights calibrated on weaker members are not guaranteed optimal for
stronger ones; the direction of that bias is toward over-weighting the analytic
baseline, which makes the resulting blend conservative rather than optimistic.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_common as S4  # noqa: E402

DATA, RESULTS, WORKFLOW, MODELS4 = S4.DATA, S4.RESULTS, S4.WORKFLOW, S4.MODELS4
SEED, CHEM_COL, log = S4.SEED, S4.CHEM_COL, S4.log

REGIMES = ("full", "chem_novel", "strain_novel", "both_novel")
SUBSAMPLE_FRAC = 0.30  # identical to the Step-3 design matrix


def selected_configs() -> dict[str, dict]:
    """Per-regime config / round count chosen by Step 3 (reused verbatim)."""
    import json

    rep = json.loads((RESULTS / "gbdt_training_report.json").read_text(encoding="utf-8"))
    sel = rep["tuning"]["selected_per_regime"]
    return {r: (sel[r]["config"], int(sel[r]["n_rounds_refit"])) for r in REGIMES}


def build_inner_design(ctx: dict, inner: dict, chunk: int = 300):
    """Assemble the inner-fit design matrix with the Step-3 cell subsampling."""
    F = ctx["F"]
    D = ctx["D"]
    enc = inner["enc"]
    fit_idx = np.flatnonzero(inner["fit_mask"])

    def cell_mask_fn(block_idx: np.ndarray) -> np.ndarray:
        """Finite-Delta cells thinned to SUBSAMPLE_FRAC, seeded per block."""
        fin = np.isfinite(D[block_idx])
        r = np.random.default_rng(SEED + int(block_idx[0]))
        return fin & (r.random(fin.shape) < SUBSAMPLE_FRAC)

    log(f"assembling the inner design matrix over {len(fit_idx)} inner_fit samples ...")
    t0 = time.time()
    X, y_abs, y_delta = F.assemble_training_matrix(
        enc, fit_idx, cell_mask_fn, chunk=chunk, label="inner_design"
    )
    log(f"  inner design {X.shape} in {time.time() - t0:.0f}s")
    assert np.isfinite(y_delta).all(), "non-finite Delta target survived the mask"
    return X, y_abs, y_delta, fit_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="retrain even if cached")
    args = ap.parse_args()

    np.random.seed(SEED)
    MODELS4.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(WORKFLOW))
    import features as F
    import lightgbm as lgb

    log("=== Step 4.4a: honest inner-dev member predictions ===")
    ctx = S4.load_context()
    inner = S4.load_inner_context(ctx)
    meta, D, C = ctx["meta"], ctx["D"], ctx["C"]
    dev_idx = inner["dev_idx"]

    # ---- analytic member: the benchmark, refitted on inner_fit -----------
    C_h_dev = ctx["C_harness"][dev_idx]
    bench_dev = np.ascontiguousarray(inner["Y_bench"][dev_idx], dtype="float32")
    d_bench = (bench_dev - C_h_dev).astype("float32")
    S4.cache_put("inner_bench_delta", d_bench)
    log(f"  benchmark delta defined on {100 * np.isfinite(d_bench).mean():.2f}% of inner-dev cells")

    need = ["inner_lgb_delta", "inner_lgb_rdkit_delta"]
    if S4.cache_has(*need) and not args.force:
        log("inner GBDT members already cached; nothing else to do")
        return

    # ---- GBDT members: trained on inner_fit only -------------------------
    X, _, y_delta, fit_idx = build_inner_design(ctx, inner)
    chem_rows = None  # filled below from row->sample mapping

    # Recover the compound name per design row. assemble_training_matrix does not
    # return the row->sample map, so rebuild it from the same cell mask: the row
    # order is sample-major, so counting kept cells per sample reproduces it.
    log("recovering the compound name per design row ...")
    counts = []
    for s in range(0, len(fit_idx), 300):
        blk = fit_idx[s : s + 300]
        fin = np.isfinite(D[blk])
        r = np.random.default_rng(SEED + int(blk[0]))
        keep = fin & (r.random(fin.shape) < SUBSAMPLE_FRAC)
        counts.append(keep.sum(axis=1))
    counts = np.concatenate(counts)
    assert int(counts.sum()) == len(X), (
        f"row-count mismatch reproducing the cell mask: {counts.sum()} vs {len(X)}; "
        "the compound-name mapping would be misaligned"
    )
    chem_rows = np.repeat(meta.iloc[fit_idx][CHEM_COL].astype(str).to_numpy(), counts)
    log(f"  mapped {len(chem_rows)} rows to {len(set(chem_rows.tolist()))} compounds")

    mol = S4.load_mol_features("gbdt")
    sel = selected_configs()

    def fit_one(Xr, cfg, n_rounds):
        params = {k: v for k, v in cfg.items() if k != "name"}
        params.update(
            {
                "verbosity": -1,
                "num_threads": 24,
                "seed": SEED,
                "deterministic": True,
                "force_row_wise": True,
                "cat_smooth": 20,
                "min_data_per_group": 100,
                "max_cat_threshold": 64,
            }
        )
        ds = lgb.Dataset(Xr, label=y_delta, categorical_feature=F.CAT_FEATURES,
                         free_raw_data=False)
        return lgb.train(params, ds, num_boost_round=n_rounds)

    models_tab: dict[str, dict] = {"inner_lgb": {}}
    models_mol: dict[str, dict] = {"inner_lgb_rdkit": {}}
    for regime in REGIMES:
        cfg, n_rounds = sel[regime]
        log(f"--- inner regime '{regime}' (rounds {n_rounds}) ---")
        Xr = F.apply_regime_mask(X, regime, copy=True)

        t0 = time.time()
        m_tab = fit_one(Xr, cfg, n_rounds)
        m_tab.save_model(str(MODELS4 / f"inner_lgb_delta__{regime}.txt"))
        log(f"  tabular fit {time.time() - t0:.0f}s")
        models_tab["inner_lgb"][regime] = (m_tab, "lgb")

        Xm = S4.attach_mol(Xr, chem_rows, mol)
        t0 = time.time()
        m_mol = fit_one(Xm, cfg, n_rounds)
        m_mol.save_model(str(MODELS4 / f"inner_lgb_rdkit_delta__{regime}.txt"))
        log(f"  rdkit fit {time.time() - t0:.0f}s ({Xm.shape[1]} features)")
        models_mol["inner_lgb_rdkit"][regime] = (m_mol, "lgb")

        del Xr, Xm

    del X
    log("predicting the inner-dev cohort with the inner specialists ...")
    raw_tab, _ = S4.predict_mol_families(
        models_tab, inner["enc"], dev_idx, meta, None, chunk=250, label="inner_tab"
    )
    S4.cache_put("inner_lgb_delta", raw_tab["inner_lgb"])

    raw_mol, regimes_dev = S4.predict_mol_families(
        models_mol, inner["enc"], dev_idx, meta, mol, chunk=250, label="inner_rdkit"
    )
    S4.cache_put("inner_lgb_rdkit_delta", raw_mol["inner_lgb_rdkit"])
    np.save(S4.cache_path("inner_regimes"), regimes_dev.astype(str))

    # routing sanity: the inner cohort was constructed by holding entities out,
    # so novelty-based routing must reproduce the regime labels we assigned.
    expected = np.array([inner["regime_of"][i] for i in dev_idx], dtype=object)
    agree = float((regimes_dev == expected).mean())
    log(f"  routing agreement with the constructed inner regimes: {100 * agree:.2f}%")

    S4.write_json(
        RESULTS / "step4_inner_members.json",
        {
            "step": "4d_inner_members",
            "seed": SEED,
            "purpose": (
                "out-of-fold member predictions on a cohort no member was fitted on, so "
                "stacking weights are not fitted on in-sample predictions"
            ),
            "n_inner_fit_samples": int(len(fit_idx)),
            "n_inner_dev_samples": int(len(dev_idx)),
            "n_design_rows": int(len(chem_rows)),
            "subsample_frac": SUBSAMPLE_FRAC,
            "regime_routing_agreement": agree,
            "members": ["inner_lgb_delta", "inner_lgb_rdkit_delta", "inner_bench_delta"],
            "inner_split_info": inner["info"],
            "caveat": (
                "inner_fit has ~2378 rows vs ~5078 for the full train split, so inner members "
                "are weaker than the final members; weights calibrated here are biased toward "
                "the analytic baseline, i.e. conservative rather than optimistic"
            ),
        },
    )
    log("=== inner members complete ===")


if __name__ == "__main__":
    main()
