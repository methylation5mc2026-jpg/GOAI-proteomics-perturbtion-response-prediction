"""Step 4.4 -- out-of-fold, per-regime, non-negative stacking meta-learner.

What is being optimised and why that shape
------------------------------------------
The competition total is a weighted sum of seven module scores, and the modules
respond to a fold-change prediction in *qualitatively different* ways:

* Module 2 (25%) is built entirely from Pearson correlations of Delta, so it is
  **invariant to any positive rescaling** of the prediction.
* Module 1 (20%) scores absolute abundance ``y = C + Delta`` and contains two
  R^2 terms, so it is **strongly scale-sensitive** -- and the Delta = 0 null
  predictor is in fact the single best Module-1 predictor found in Step 3
  (0.8415, above the 0.4454 benchmark's 0.8094).
* Module 4 (5%) thresholds ``|Delta| > 1``, so it is scale-sensitive too.
* The Module-3 residual terms subtract a fixed baseline from both sides and are
  therefore neither invariant nor monotone in the scale.

A single member cannot sit at the optimum of that mixture. A **non-negative
linear combination** can, because the shortfall of the weights from 1 acts as a
shrinkage toward the control anchor: it buys Module 1 and Module 4 at no cost to
Module 2. This is not a heuristic -- it is the structure of the rubric, and it is
why the meta-learner is given weights that need not sum to one and are not
constrained to a simplex.

Weights are fitted **per availability regime**, because the regime is knowable at
submission time from entity novelty alone (Step 3 proved that rule reproduces the
four official splits exactly), and because the right mixture genuinely differs by
regime: the benchmark is strong where a novel chemical destroys the learned
features and weak where it does not.

Where the weights are fitted -- the part that makes the number honest
-------------------------------------------------------------------
Fitting weights on ``val_*`` and then reporting the ``val_*`` score would make the
headline a tuned-on number, which is precisely the trap Step 3 refused to walk
into when it left ``BLEND_W`` pre-specified at 0.5. So:

* weights are fitted **only** on the inner-dev cohort carved out of ``train``,
  using members retrained on ``inner_fit`` (``25a_inner_members.py``);
* the fitted weights are then **frozen** and applied to ``val_*`` once;
* the ``val_*`` score reported is therefore a genuine held-out estimate.

For transparency the script *also* reports the val-optimal weights as an
explicitly-labelled optimistic upper bound. That number is a diagnostic of how
much headroom the weight space contains -- it is never presented as the result.

Outputs
-------
results/step4_model_scores.json     all candidates, per-module, both cohorts
results/step4_stacking_weights.json the frozen weights actually used
figures/step4_performance_comparison.png
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

DATA, RESULTS, FIGURES, WORKFLOW = S4.DATA, S4.RESULTS, S4.FIGURES, S4.WORKFLOW
SEED, log = S4.SEED, S4.log

REGIMES = ("full", "chem_novel", "strain_novel", "both_novel")

#: Member roles. Each role must resolve to a cached matrix on BOTH cohorts, and
#: the two must be produced by the same library at the same hyper-parameters --
#: only the fit set may differ. LightGBM is used for the learned roles because it
#: is the one library for which an inner-fit counterpart was trained, so the
#: inner/val correspondence is exact. Substituting the stronger multi-library
#: ensemble for the val side would break that correspondence, and is reported
#: separately as a labelled sensitivity instead.
ROLES = {
    "gbdt_tab": ("inner_lgb_delta", "val_lgb_delta"),
    "gbdt_mol": ("inner_lgb_rdkit_delta", "val_lgb_rdkit_delta"),
    "bench": ("inner_bench_delta", "val_bench_delta"),
    "dl": ("dl_delta_inner", "dl_delta_val"),
}

BENCH_TOTAL = 0.445442994  # the Step-2/3 benchmark this step must beat


# ---------------------------------------------------------------------------
# Cohort assembly
# ---------------------------------------------------------------------------
def load_cohort(which: str, ctx: dict, inner: dict | None) -> dict:
    """Gather targets, baselines, members and regime labels for one cohort."""
    meta, Y, D = ctx["meta"], ctx["Y"], ctx["D"]
    slot = 0 if which == "inner" else 1

    members: dict[str, np.ndarray] = {}
    for role, names in ROLES.items():
        arr = S4.cache_get(names[slot])
        if arr is None:
            log(f"  !! role '{role}' missing on {which} ({names[slot]}.npy absent); dropped")
            continue
        members[role] = arr

    if which == "inner":
        idx = inner["dev_idx"]
        meta_eval = inner["meta_dev"]
        mu_ctx, mu_drug = inner["mu_ctx"][idx], inner["mu_drug"][idx]
        y_fb = inner["y_fallback"][idx]
        regimes = np.array([inner["regime_of"][i] for i in idx], dtype=object)
    else:
        mask = ctx["masks"]["all_val"]
        idx = np.flatnonzero(mask)
        meta_eval = meta.loc[mask].reset_index(drop=True)
        mu_ctx, mu_drug = ctx["mu_ctx"][mask], ctx["mu_drug"][mask]
        y_fb = S4.cache_get("val_y_fallback")
        if y_fb is None:
            y_fb = ctx["y_fallback"][mask]
        rp = S4.cache_path("val_regimes")
        regimes = (
            np.load(rp, allow_pickle=False).astype(object)
            if rp.exists()
            else S4.regimes_for_samples(ctx["enc"], meta, idx)
        )

    C_h = ctx["C_harness"][idx]
    # Members are finite except the analytic benchmark, which is undefined where
    # its context group had too few observations. An undefined component
    # contributes zero in Delta space, i.e. that cell falls back to the control
    # anchor for that member -- the same convention Step 3 used.
    for role in members:
        members[role] = np.nan_to_num(members[role], nan=0.0, posinf=0.0, neginf=0.0)

    log(
        f"  cohort '{which}': {len(idx)} samples, roles {sorted(members)}, "
        f"regimes {dict(pd.Series(regimes).value_counts())}"
    )
    return {
        "idx": idx,
        "meta_eval": meta_eval,
        "Y": Y[idx],
        "D": D[idx],
        "C_h": C_h,
        "y_fb": y_fb,
        "mu_ctx": mu_ctx,
        "mu_drug": mu_drug,
        "members": members,
        "regimes": regimes,
        "roles": sorted(members),
    }


# ---------------------------------------------------------------------------
# Blending and scoring
# ---------------------------------------------------------------------------
def blend(coh: dict, W: np.ndarray, roles: list[str]) -> np.ndarray:
    """Combine members with per-regime non-negative weights.

    ``W`` has shape ``(n_regimes, n_roles)``; row order follows :data:`REGIMES`.
    """
    out = np.zeros_like(coh["C_h"])
    for ri, regime in enumerate(REGIMES):
        rows = np.flatnonzero(coh["regimes"] == regime)
        if not len(rows):
            continue
        acc = np.zeros((len(rows), out.shape[1]), dtype="float32")
        for ki, role in enumerate(roles):
            w = float(W[ri, ki])
            if w != 0.0:
                acc += w * coh["members"][role][rows]
        out[rows] = acc
    return out


def score_blend(coh: dict, W: np.ndarray, roles: list[str]) -> dict:
    """Full harness score for a weight matrix on one cohort."""
    d = blend(coh, W, roles)
    yh, dh = S4.reconstruct(d, coh["C_h"], coh["y_fb"])
    return S4.score(
        coh["Y"], yh, coh["D"], dh, coh["meta_eval"], coh["mu_ctx"], coh["mu_drug"]
    )


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
def optimise(coh: dict, roles: list[str], fast, m4: float, n_restarts: int = 6,
             maxiter: int = 25, seed: int = SEED, time_budget: float = 600.0,
             ) -> tuple[np.ndarray, float, list[dict]]:
    """Maximise the harness total over non-negative per-regime weights.

    Coordinate ascent on a coarse-to-fine grid rather than a general-purpose
    optimiser: each objective evaluation costs a full harness pass over the
    cohort, the search space is only ``n_regimes x n_roles``, and the objective is
    piecewise-smooth but not differentiable (Module 4 thresholds, ``clip01``).
    Coordinate ascent needs far fewer evaluations than Nelder-Mead here and never
    proposes a negative weight.

    The objective is :class:`step4_fastscore.FastScorer`, which reproduces
    modules 1-3 (0.95 of the total weight) exactly but in microseconds rather
    than the ~20 s an exact harness pass costs. Module 4 is not a polynomial in
    the weights, so it is held at the constant ``m4`` during the search and every
    shortlisted candidate is re-scored with the real harness afterwards -- no
    reported number comes from the fast path.

    ``time_budget`` caps the wall-clock spend. If the cap is hit the search stops
    at the best point found so far, and how much of the schedule was skipped is
    logged -- a silently truncated search would otherwise be indistinguishable
    from a converged one.
    """
    nR, nK = len(REGIMES), len(roles)
    rng = np.random.default_rng(seed)
    trace: list[dict] = []
    n_eval = 0
    t_start = time.time()
    truncated = False

    def obj(W):
        nonlocal n_eval
        n_eval += 1
        return fast.total(W, m4=m4)

    def out_of_time() -> bool:
        nonlocal truncated
        if time.time() - t_start > time_budget:
            truncated = True
            return True
        return False

    best_W, best_s = None, -np.inf
    starts = [np.full((nR, nK), 1.0 / max(nK, 1), dtype=float)]
    # A start that leans on the analytic baseline, and a shrunken one, because
    # the objective is not concave and the basin matters.
    if "bench" in roles:
        w0 = np.zeros((nR, nK))
        w0[:, roles.index("bench")] = 1.0
        starts.append(w0)
    starts.append(np.full((nR, nK), 0.25, dtype=float))
    while len(starts) < n_restarts:
        starts.append(rng.uniform(0.0, 0.8, size=(nR, nK)))

    grids = [
        np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25]),
        np.array([-0.08, -0.04, -0.02, 0.0, 0.02, 0.04, 0.08]),
        np.array([-0.015, -0.008, -0.004, 0.0, 0.004, 0.008, 0.015]),
    ]

    for si, W0 in enumerate(starts[:n_restarts]):
        if out_of_time():
            break
        W = W0.copy()
        s = obj(W)
        log(f"  [opt] start {si}: total={s:.6f}")
        for gi, grid in enumerate(grids):
            improved = True
            sweep = 0
            while improved and sweep < maxiter and not out_of_time():
                improved = False
                sweep += 1
                for ri in range(nR):
                    for ki in range(nK):
                        base = W[ri, ki]
                        cand = np.clip(grid if gi == 0 else base + grid, 0.0, None)
                        for v in cand:
                            if v == base:
                                continue
                            W[ri, ki] = v
                            sv = obj(W)
                            if sv > s + 1e-9:
                                s, base, improved = sv, v, True
                            else:
                                W[ri, ki] = base
                    if out_of_time():
                        break
                log(f"  [opt] start {si} grid {gi} sweep {sweep}: total={s:.6f} "
                    f"({n_eval} evals, {time.time() - t_start:.0f}s)")
        trace.append({"start": si, "total": s, "weights": W.tolist()})
        if s > best_s:
            best_W, best_s = W.copy(), s
    log(f"  [opt] best total={best_s:.6f} after {n_eval} evaluations "
        f"in {time.time() - t_start:.0f}s"
        + (f" -- SEARCH TRUNCATED at the {time_budget:.0f}s budget, "
           f"{len(trace)}/{n_restarts} restarts completed" if truncated else ""))
    return best_W, best_s, trace


# ---------------------------------------------------------------------------
def weights_table(W: np.ndarray, roles: list[str]) -> dict:
    """Human-readable nested dict of the weight matrix."""
    return {
        regime: {role: round(float(W[ri, ki]), 4) for ki, role in enumerate(roles)}
        for ri, regime in enumerate(REGIMES)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=3)
    args = ap.parse_args()

    np.random.seed(SEED)
    log("=== Step 4.4: out-of-fold stacking meta-learner ===")

    ctx = S4.load_context()
    inner = S4.load_inner_context(ctx)

    log("assembling cohorts ...")
    coh_in = load_cohort("inner", ctx, inner)
    coh_val = load_cohort("val", ctx, inner)
    roles = [r for r in ROLES if r in coh_in["roles"] and r in coh_val["roles"]]
    log(f"roles usable on both cohorts: {roles}")
    if not roles:
        raise SystemExit("no member role is available on both cohorts")

    t0 = time.time()
    _ = score_blend(coh_in, np.zeros((len(REGIMES), len(roles))), roles)
    exact_cost = time.time() - t0
    log(f"one exact harness evaluation on the inner cohort costs {exact_cost:.2f}s")

    # ---- fast exact objective for modules 1-3, validated before use ------
    import step4_fastscore as FS

    log("building the fast co-moment objective for the inner cohort ...")
    t0 = time.time()
    fast_in = FS.FastScorer(coh_in, roles, REGIMES)
    log(f"  built in {time.time() - t0:.1f}s")
    val_in = fast_in.validate(
        coh_in, lambda W: score_blend(coh_in, W, roles), n_trials=3, seed=SEED
    )

    log("building the fast co-moment objective for the validation cohort ...")
    t0 = time.time()
    fast_val = FS.FastScorer(coh_val, roles, REGIMES)
    log(f"  built in {time.time() - t0:.1f}s")
    val_val = fast_val.validate(
        coh_val, lambda W: score_blend(coh_val, W, roles), n_trials=3, seed=SEED + 1
    )

    # ---- individual members, both cohorts -------------------------------
    results: dict[str, dict] = {}
    for name, coh in (("inner", coh_in), ("val", coh_val)):
        for ki, role in enumerate(roles):
            W = np.zeros((len(REGIMES), len(roles)))
            W[:, ki] = 1.0
            res = score_blend(coh, W, roles)
            results[f"{name}:{role}"] = res
            log(f"  {name:5s} {role:10s} total={res['total_score']:.6f}")
        W0 = np.zeros((len(REGIMES), len(roles)))
        res = score_blend(coh, W0, roles)
        results[f"{name}:control_anchor"] = res
        log(f"  {name:5s} {'zero':10s} total={res['total_score']:.6f}")

    # ---- fit the weights on the inner cohort ONLY ------------------------
    # Module 4 is held at the value it takes for the equal-weight start; the
    # shortlist is re-scored exactly afterwards, so this only affects which
    # candidates are proposed, never which number is reported.
    m4_in = float(
        results["inner:" + roles[0]]["module_scores"]["m4_dep"]
    )
    log("fitting stacking weights on the inner-dev cohort (val is never touched) ...")
    W_in, s_in_fast, trace = optimise(
        coh_in, roles, fast_in, m4_in, n_restarts=args.restarts
    )
    log(f"inner-dev optimum (fast objective): {s_in_fast:.6f}")

    # Re-score the shortlist with the REAL harness and pick by the exact total.
    shortlist = [W_in] + [np.array(t["weights"]) for t in trace]
    seen, cands = set(), []
    for Wc in shortlist:
        key = np.round(Wc, 4).tobytes()
        if key not in seen:
            seen.add(key)
            cands.append(Wc)
    log(f"re-scoring {len(cands)} shortlisted candidates with the exact harness ...")
    exact = []
    for i, Wc in enumerate(cands):
        r = score_blend(coh_in, Wc, roles)
        exact.append((float(r["total_score"]), Wc))
        log(f"  candidate {i}: fast-selected -> exact inner total {r['total_score']:.6f}")
    exact.sort(key=lambda kv: -kv[0])
    s_in, W_in = exact[0][0], exact[0][1]
    log(f"inner-dev optimum (exact harness): {s_in:.6f}")
    log("frozen weights:\n" + json.dumps(weights_table(W_in, roles), indent=2))

    # ---- apply the frozen weights to val: the headline number -----------
    res_val = score_blend(coh_val, W_in, roles)
    results["val:stacked_oof_frozen"] = res_val
    log(f"*** val total with frozen OOF weights = {res_val['total_score']:.6f} "
        f"(benchmark {BENCH_TOTAL:.6f}) ***")

    results["inner:stacked_oof_frozen"] = score_blend(coh_in, W_in, roles)

    # ---- diagnostic only: the val-optimal weights (optimistic) ----------
    log("diagnostic: fitting weights directly on val to measure available headroom "
        "(OPTIMISTIC, never reported as the result) ...")
    m4_val = float(results["val:" + roles[0]]["module_scores"]["m4_dep"])
    W_valopt, _, trace_v = optimise(
        coh_val, roles, fast_val, m4_val, n_restarts=max(2, args.restarts // 2)
    )
    cand_v = [W_valopt] + [np.array(t["weights"]) for t in trace_v]
    best_v = max(
        ((float(score_blend(coh_val, Wc, roles)["total_score"]), Wc) for Wc in cand_v),
        key=lambda kv: kv[0],
    )
    s_val, W_val = best_v
    results["val:stacked_val_tuned_OPTIMISTIC"] = score_blend(coh_val, W_val, roles)

    # ---- pre-specified sensitivity: multi-library GBDT in the tabular role
    sens: dict = {}
    ens_parts = [
        S4.cache_get(f"val_{lib}_delta") for lib in ("lgb", "xgb", "cat")
    ]
    ens_parts = [a for a in ens_parts if a is not None]
    if len(ens_parts) > 1:
        coh_alt = dict(coh_val)
        coh_alt["members"] = dict(coh_val["members"])
        coh_alt["members"]["gbdt_tab"] = np.mean(ens_parts, axis=0).astype("float32")
        r = score_blend(coh_alt, W_in, roles)
        sens["val_gbdt_tab_replaced_by_multi_library_ensemble"] = {
            "total_score": r["total_score"],
            "module_scores": r["module_scores"],
            "note": (
                f"the tabular role is replaced by the mean of {len(ens_parts)} libraries, which "
                "is stronger than the LightGBM member the weights were calibrated against; "
                "reported as a sensitivity because the inner/val member correspondence is no "
                "longer exact"
            ),
        }
        results["val:stacked_frozen_multilib"] = r
        log(f"  sensitivity (multi-library tabular role): {r['total_score']:.6f}")

    rdk = S4.cache_get("val_xgb_rdkit_delta")
    if rdk is not None and "gbdt_mol" in roles:
        coh_alt = dict(coh_val)
        coh_alt["members"] = dict(coh_val["members"])
        coh_alt["members"]["gbdt_mol"] = np.nan_to_num(
            (coh_val["members"]["gbdt_mol"] + rdk) / 2.0, nan=0.0
        ).astype("float32")
        r = score_blend(coh_alt, W_in, roles)
        sens["val_gbdt_mol_averaged_with_xgb_rdkit"] = {
            "total_score": r["total_score"],
            "module_scores": r["module_scores"],
        }
        results["val:stacked_frozen_molens"] = r
        log(f"  sensitivity (lgb+xgb RDKit role): {r['total_score']:.6f}")

    # ---- report ----------------------------------------------------------
    headline = float(res_val["total_score"])
    out = {
        "step": "4d_stacking",
        "seed": SEED,
        "benchmark_total": BENCH_TOTAL,
        "step3_best_total": 0.436172,
        "roles": roles,
        "role_sources": {r: {"inner": ROLES[r][0], "val": ROLES[r][1]} for r in roles},
        "protocol": {
            "weight_space": (
                "non-negative, per-regime, NOT constrained to sum to one: the shortfall from "
                "one is a shrinkage toward the control anchor, which buys Module 1 and Module 4 "
                "without costing Module 2 (whose submetrics are scale-invariant Pearson terms)"
            ),
            "fitted_on": (
                "inner-dev cohort carved from train, with every member retrained on inner_fit "
                "and both residual baselines refitted on inner_fit"
            ),
            "evaluated_on": "val_* with the inner-fitted weights frozen",
            "regime_routing": "entity novelty (transfers to the test set unchanged)",
            "optimiser": "coordinate ascent, coarse-to-fine grid, non-negativity by construction",
            "objective_implementation": (
                "modules 1-3 (0.95 of the weight) are evaluated by a co-moment "
                "factorisation that is algebraically identical to the harness but ~10^5 times "
                "faster, which is what makes a converged search possible at all; module 4 is "
                "not a polynomial in the weights so it is held constant during the search and "
                "every shortlisted candidate is re-scored with the real harness, from which "
                "the reported total is taken"
            ),
            "fast_objective_validation": {
                "inner": {k: v for k, v in val_in.items() if k != "rows"},
                "val": {k: v for k, v in val_val.items() if k != "rows"},
                "note": "max absolute deviation of fast vs exact module scores; gate is 1e-6",
            },
            "exact_harness_seconds_per_evaluation": round(exact_cost, 2),
        },
        "frozen_weights": weights_table(W_in, roles),
        "inner_dev_total_at_frozen_weights": float(s_in),
        "val_total_at_frozen_weights": headline,
        "val_total_val_tuned_OPTIMISTIC": float(s_val),
        "generalisation_gap_inner_to_val": float(headline - s_in),
        "beats_benchmark": bool(headline > BENCH_TOTAL),
        "margin_vs_benchmark": float(headline - BENCH_TOTAL),
        "optimiser_trace": trace,
        "sensitivity": sens,
        "module_scores": {k: v["module_scores"] for k, v in results.items()},
        "module_weights": next(iter(results.values()))["module_weights"],
        "totals": {k: float(v["total_score"]) for k, v in results.items()},
        "caveats": [
            "inner_fit (~2378 rows) is smaller than train (~5078), so inner members are weaker "
            "than the final members; weights calibrated on them are biased toward the analytic "
            "baseline, which is conservative rather than optimistic",
            "the val-tuned total is reported only to quantify headroom in the weight space and "
            "must not be read as an achieved score",
        ],
    }
    S4.write_json(RESULTS / "step4_stacking_weights.json", out)
    S4.write_json(RESULTS / "step4_model_scores.json", out)

    print("\n=== TOTALS ===")
    for k, v in sorted(out["totals"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:45s} {v:.6f}")
    print(f"\n  benchmark to beat: {BENCH_TOTAL:.6f}")
    print(f"  HEADLINE (frozen OOF weights on val): {headline:.6f} "
          f"({headline - BENCH_TOTAL:+.6f})")
    log("=== stacking complete ===")


if __name__ == "__main__":
    main()
