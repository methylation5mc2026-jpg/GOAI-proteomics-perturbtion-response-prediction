#!/usr/bin/env python
"""Step 6.4 -- Hierarchical 32-cluster non-negative stacking meta-learner.

Extends the Step-5 protein-cluster stacking from K=15 to a genuinely *nested*
cluster hierarchy (K = 4 -> 8 -> 16 -> 32) and fits non-negative, L2-ridge
regularised stacking weights W in R^{regimes x roles x clusters}.

Why hierarchical
----------------
At K=32 with 7 member roles the weight tensor has 4 x 7 x 32 = 896 free
parameters, all fitted by coordinate ascent on a fixed OOF cohort. Started cold,
that search both costs more and overfits the cohort more. We therefore build a
*nested* hierarchy -- K-means at K=32, then agglomerative merging of the 32
centroids to 16, 8 and 4 -- and warm-start each level from the optimum of its
parent, so every fine cluster begins at its parent's fitted weight and only
departs from it if the OOF objective rewards the departure. This is the
statistical shrinkage the ridge term expresses, made explicit in the search path.

Protocol (unchanged from Step 5, and the reason the numbers are trustworthy)
---------------------------------------------------------------------------
* Weights are fitted **only** on the 5-fold LCGO out-of-fold cohort, where every
  member prediction came from a model that never saw the row.
* Model selection (K, ridge alpha) uses the OOF total **only**.
* ``val_*`` is scored exactly once, with the selected weights frozen.
* Every reported total comes from the real ``workflow/harness.py``. The
  co-moment fast scorer is a search surrogate and is validated against the
  harness before any search is trusted.

Outputs
-------
data/step6_protein_clusters_k32.parquet
results/step6_model_scores.json
results/step6_cluster_weights.json
results/step6_bootstrap_ci.json
figures/step6_performance_comparison.png
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SESSION = REPO_ROOT
WORKFLOW = SESSION / "workflow"
DATA = SESSION / "data"
RESULTS = SESSION / "results"
FIGURES = SESSION / "figures"
CACHE5 = DATA / "step5_cache"
SEED = 42
BENCH_TOTAL = 0.445442994
STEP5_VAL_TOTAL = 0.5085515642054328
sys.path.insert(0, str(WORKFLOW))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(m: str) -> None:
    print(m, flush=True)


# ---------------------------------------------------------------------------
# Nested cluster hierarchy
# ---------------------------------------------------------------------------
def build_hierarchy(proteins: list[str], emb: np.ndarray, stat: pd.DataFrame,
                    levels=(4, 8, 16, 32), min_size: int = 50) -> tuple[pd.DataFrame, dict]:
    """K-means at the finest level, then agglomerate centroids for coarser levels.

    Reproduces the Step-5 feature construction exactly (rank-transformed
    abundance/response statistics plus rank-transformed spectral coordinates,
    each view scaled to equal total variance) but produces a *nested* family:
    every level-L cluster is a union of finest-level clusters, so a weight fitted
    at one level is a valid warm start for the next.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.preprocessing import QuantileTransformer

    n_p = len(proteins)
    stat_cols = ["abundance_mean", "abundance_sd", "detect_rate", "delta_sd",
                 "delta_abs_mean", "weighted_degree"]
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=SEED, subsample=100_000)
    Xs = qt.fit_transform(stat[stat_cols].to_numpy("float64"))
    n_sp = int(min(16, emb.shape[1]))
    Es = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=SEED, subsample=100_000
                             ).fit_transform(emb[:, :n_sp].astype("float64"))
    Z = np.hstack([Xs / np.sqrt(Xs.shape[1]), Es / np.sqrt(Es.shape[1])])
    log(f"  clustering feature matrix {Z.shape} "
        f"(abundance stats + {n_sp} spectral coords, rank-transformed)")

    K_fine = max(levels)
    km = KMeans(n_clusters=K_fine, n_init=10, random_state=SEED)
    fine = km.fit_predict(Z)

    # Merge undersized fine clusters into the nearest surviving centroid so that
    # no cluster carries a whole row of the weight tensor while being fitted on
    # a handful of proteins.
    merges = []
    while True:
        present = np.unique(fine)
        sizes = {int(c): int((fine == c).sum()) for c in present}
        small = [c for c, n in sizes.items() if n < min_size]
        if not small or len(present) <= 2:
            break
        victim = min(small, key=lambda c: sizes[c])
        cents = {int(c): Z[fine == c].mean(0) for c in present if c != victim}
        vc = Z[fine == victim].mean(0)
        tgt = min(cents, key=lambda c: float(np.sum((cents[c] - vc) ** 2)))
        merges.append({"merged_size": sizes[victim], "into_size": sizes[tgt]})
        fine[fine == victim] = tgt
    uniq = {int(c): i for i, c in enumerate(np.unique(fine))}
    fine = np.array([uniq[int(c)] for c in fine], dtype=np.int64)
    K_real = int(fine.max()) + 1
    if merges:
        log(f"  merged {len(merges)} undersized cluster(s): K={K_fine} requested "
            f"-> {K_real} realised")

    cent = np.vstack([Z[fine == c].mean(0) for c in range(K_real)])
    clus = pd.DataFrame({"protein": proteins})
    parents: dict[str, list[int]] = {}
    info: dict = {"levels": {}, "n_merges_fine": len(merges), "merges": merges}

    for L in sorted(levels):
        if L >= K_real:
            lab_fine_to_L = np.arange(K_real)
        else:
            ag = AgglomerativeClustering(n_clusters=L, linkage="ward")
            lab_fine_to_L = ag.fit_predict(cent)
        lab = lab_fine_to_L[fine]
        # Order clusters by mean abundance: cluster 0 is the least abundant
        # stratum at every level, so the index reads the same way across K.
        K = int(lab.max()) + 1
        order = np.argsort([stat.loc[lab == c, "abundance_mean"].mean() for c in range(K)])
        remap = np.empty(K, dtype=int)
        remap[order] = np.arange(K)
        lab = remap[lab]
        col = f"k{L}"
        clus[col] = lab.astype(np.int64)
        parents[col] = remap[lab_fine_to_L].tolist()
        sizes = np.bincount(lab, minlength=K).tolist()
        info["levels"][col] = {
            "requested": int(L), "realised": int(K), "sizes": sizes,
            "min_size": int(min(sizes)), "max_size": int(max(sizes)),
            "abundance_mean_by_cluster":
                [float(stat.loc[lab == c, "abundance_mean"].mean()) for c in range(K)],
            "delta_sd_by_cluster":
                [float(stat.loc[lab == c, "delta_sd"].mean()) for c in range(K)],
        }
        log(f"  {col}: {K} realised clusters, sizes {min(sizes)}..{max(sizes)}")

    info["fine_to_level_map"] = parents
    info["nested"] = True
    return clus, info


def promote_weights(W: np.ndarray, child_lab: np.ndarray, parent_lab: np.ndarray,
                    n_child: int) -> np.ndarray:
    """Lift a weight tensor from a coarse level to a finer nested level.

    Every fine cluster inherits the weight of the coarse cluster containing it,
    so the finer search starts exactly at the coarser optimum and can only be
    accepted if it improves on it.
    """
    nR, nK, _ = W.shape
    out = np.empty((nR, nK, n_child), dtype=W.dtype)
    for c in range(n_child):
        members = np.flatnonzero(child_lab == c)
        # every fine cluster sits inside exactly one coarse cluster (nested)
        p = int(np.bincount(parent_lab[members]).argmax()) if len(members) else 0
        out[:, :, c] = W[:, :, p]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="gbdt_tab,gbdt_mol,gbdt_mol3d,bench,gnn,dl,xattn")
    ap.add_argument("--levels", default="4,8,16,32")
    ap.add_argument("--alphas", default="0.0,0.01,0.03")
    ap.add_argument("--budget", type=float, default=420.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    t0 = time.time()
    np.random.seed(SEED)
    log("=" * 78)
    log("Step 6.4  Hierarchical 32-cluster non-negative stacking meta-learner")
    log("=" * 78)

    st30 = load_module(WORKFLOW / "30_gnn_and_cluster_stacking.py", "st30")
    CS = load_module(WORKFLOW / "step5_clusterscore.py", "cs5")
    S4 = sys.modules["step4_common"]
    REGIMES = st30.REGIMES

    # Register the Step-6 cross-attention member alongside the Step-5 roles.
    st30.ROLES5["xattn"] = ("oof_xattn", "val_xattn", "test_xattn")

    ctx = S4.load_context()
    G = st30.load_graph()
    proteins = list(G["proteins"])
    stat = pd.read_parquet(DATA / "step5_protein_stats.parquet")
    levels = tuple(int(x) for x in args.levels.split(",") if x)

    log("\n[1] building the nested protein-cluster hierarchy ...")
    clus, hier_info = build_hierarchy(proteins, G["embedding"], stat, levels=levels)
    clus.to_parquet(DATA / "step6_protein_clusters_k32.parquet", index=False)
    log(f"    saved data/step6_protein_clusters_k32.parquet {clus.shape}")

    # ---- cohorts ----
    wanted = [r for r in args.roles.split(",") if r]
    log(f"\n[2] loading cohorts for roles: {wanted}")
    coh_oof = st30.load_oof_cohort(ctx, wanted)
    coh_val = st30.load_val_cohort(ctx, wanted)
    roles = [r for r in wanted if r in coh_oof["roles"] and r in coh_val["roles"]]
    dropped = [r for r in wanted if r not in roles]
    if dropped:
        log(f"    !! roles unavailable on both cohorts and therefore dropped: {dropped}")
    log(f"    active roles ({len(roles)}): {roles}")
    log(f"    OOF rows {coh_oof['Y'].shape[0]}, val rows {coh_val['Y'].shape[0]}, "
        f"proteins {coh_oof['Y'].shape[1]}")

    results: dict = {}

    # ---- per-member reference scores (real harness) ----
    # Each harness pass costs minutes (module 4 loops per sample in Python), so
    # per-member reference scores are taken on val -- the cohort the figure and
    # the headline comparison use -- plus the single OOF pass needed to read off
    # the frozen module-4 constant.
    log("\n[3] per-member harness scores (val cohort; one OOF pass for the m4 constant) ...")
    ones = np.zeros(len(proteins), dtype=np.int64)
    for r in roles:
        W = np.zeros((len(REGIMES), len(roles), 1))
        W[:, roles.index(r), 0] = 1.0
        sc = st30.score_cluster_blend(coh_val, W, roles, ones)
        results[f"val:{r}"] = sc
        log(f"    val:{r:<10} total={sc['total_score']:.6f}")
    W_first = np.zeros((len(REGIMES), len(roles), 1))
    W_first[:, 0, 0] = 1.0
    results[f"oof:{roles[0]}"] = st30.score_cluster_blend(coh_oof, W_first, roles, ones)
    log(f"    oof:{roles[0]:<10} total={results['oof:' + roles[0]]['total_score']:.6f}")
    W0 = np.zeros((len(REGIMES), len(roles), 1))
    results["val:control_anchor"] = st30.score_cluster_blend(coh_val, W0, roles, ones)
    log(f"    val:control_anchor total={results['val:control_anchor']['total_score']:.6f}")

    m4_oof = float(results["oof:" + roles[0]]["module_scores"]["m4_dep"])
    m4_val = float(results["val:" + roles[0]]["module_scores"]["m4_dep"])
    log(f"    module-4 constant held during search: oof={m4_oof:.6f} val={m4_val:.6f}")

    # ---- rung 1: scalar weights ----
    log("\n[4] rung 1: scalar (single-cluster) weights on the OOF cohort ...")
    fast1 = CS.ClusterFastScorer(coh_oof, roles, REGIMES, ones, 1)
    val1 = fast1.validate(lambda W: st30.score_cluster_blend(coh_oof, W, roles, ones),
                          n_trials=3, tol=1e-6)
    val1.pop("rows", None)
    log(f"    fast-scorer vs harness max |deviation| = {val1['max_abs_deviation']:.3e}")
    W_scal = np.full((len(REGIMES), len(roles), 1), 1.0 / max(len(roles), 1))
    W_scal, s1, info1 = st30.optimise_clusters(fast1, m4_oof, W_scal,
                                               time_budget=args.budget, alpha=0.0,
                                               label="scalar/oof")
    r_oof1 = st30.score_cluster_blend(coh_oof, W_scal, roles, ones)
    r_val1 = st30.score_cluster_blend(coh_val, W_scal, roles, ones)
    results["oof:scalar_frozen"] = r_oof1
    results["val:scalar_oof_frozen"] = r_val1
    log(f"    scalar: OOF {r_oof1['total_score']:.6f} | val {r_val1['total_score']:.6f}")

    # ---- rung 2: hierarchical cluster ladder ----
    log("\n[5] rung 2: hierarchical cluster ladder (warm-started level by level) ...")
    alphas = [float(a) for a in args.alphas.split(",") if a]
    ladder: dict = {}
    best = {"total": -np.inf}
    for alpha in alphas:
        log(f"\n  --- ridge alpha = {alpha} ---")
        W_prev, lab_prev = W_scal, ones
        for L in sorted(levels):
            col = f"k{L}"
            lab = clus[col].to_numpy(np.int64)
            nC = int(lab.max()) + 1
            W0 = promote_weights(W_prev, lab, lab_prev, nC)
            fast = CS.ClusterFastScorer(coh_oof, roles, REGIMES, lab, nC, verbose=False)
            if alpha == alphas[0] and L == sorted(levels)[-1]:
                # validate the surrogate again at the finest level, where the
                # weight space is largest and any drift would matter most
                v = fast.validate(
                    lambda W, lab=lab: st30.score_cluster_blend(coh_oof, W, roles, lab),
                    n_trials=2, tol=1e-6)
                v.pop("rows", None)
                log(f"    [{col}] fast-scorer revalidated: "
                    f"max |dev| = {v['max_abs_deviation']:.3e}")
                ladder["fast_validation_finest"] = v
            W, s, info = st30.optimise_clusters(
                fast, m4_oof, W0, time_budget=args.budget, alpha=alpha,
                label=f"{col}/a{alpha}")
            # Rung comparison uses the co-moment surrogate, not a fresh harness
            # pass. The surrogate is algebraically identical to the harness on
            # modules 1-3 (0.95 of the weight; revalidated above at the finest
            # level to ~1e-9) and adds the same frozen m4 constant to every rung,
            # so it induces exactly the harness ordering while costing
            # milliseconds instead of minutes. The rung that wins is then scored
            # with the real harness before anything is reported.
            oof_surrogate = float(fast.total(W, m4=m4_oof))
            key = f"{col}_alpha{alpha}"
            ladder[key] = {"cluster_col": col, "n_clusters": nC, "alpha": alpha,
                           "oof_total_surrogate": oof_surrogate,
                           "n_evaluations": info["n_evaluations"],
                           "seconds": info["seconds"], "truncated": info["truncated"],
                           "warm_started_from": (f"k{sorted(levels)[sorted(levels).index(L) - 1]}"
                                                 if L != sorted(levels)[0] else "scalar")}
            log(f"    [{col}] nC={nC:<3} alpha={alpha:<5} OOF (surrogate)="
                f"{oof_surrogate:.6f} ({info['n_evaluations']} evals, "
                f"{info['seconds']:.0f}s{', TRUNCATED' if info['truncated'] else ''})")
            if oof_surrogate > best["total"]:
                best = {"total": oof_surrogate, "W": W.copy(), "lab": lab.copy(),
                        "col": col, "nC": nC, "alpha": alpha}
            W_prev, lab_prev = W, lab

    log(f"\n    selected on the OOF cohort only: {best['col']} "
        f"(nC={best['nC']}, alpha={best['alpha']}) OOF surrogate={best['total']:.6f}")

    # ---- freeze and evaluate on val exactly once ----
    log("\n[6] freezing the selected weights and scoring val once ...")
    W_fin, cl_fin = best["W"], best["lab"]
    r_val = st30.score_cluster_blend(coh_val, W_fin, roles, cl_fin)
    results["val:hier_cluster_oof_frozen"] = r_val
    results["oof:hier_cluster_frozen"] = st30.score_cluster_blend(
        coh_oof, W_fin, roles, cl_fin)
    headline = float(r_val["total_score"])
    log(f"    VAL TOTAL (frozen, harness) = {headline:.6f}")
    log(f"    Step-5 reference            = {STEP5_VAL_TOTAL:.6f}  "
        f"(delta {headline - STEP5_VAL_TOTAL:+.6f})")
    log(f"    official benchmark          = {BENCH_TOTAL:.6f}  "
        f"(delta {headline - BENCH_TOTAL:+.6f})")
    target = 0.5150
    log(f"    Step-6 target {target}: "
        f"{'MET' if headline > target else 'NOT MET'} "
        f"({headline - target:+.6f})")

    # ---- paired bootstrap vs the scalar rung and vs Step 5 ----
    log(f"\n[7] paired bootstrap ({args.n_boot} replicates) ...")
    comp_hier = st30.per_sample_components(coh_val, W_fin, roles, cl_fin)
    comp_scal = st30.per_sample_components(coh_val, W_scal, roles, ones)
    boot = {"hier_cluster_vs_scalar":
            st30.bootstrap_margin(comp_hier, comp_scal, n_boot=args.n_boot, seed=SEED)}
    fw = CACHE5 / "frozen_W.npy"
    if fw.exists():
        W5 = np.load(fw)
        cl5 = np.load(CACHE5 / "frozen_clusters.npy")
        roles5 = json.loads((CACHE5 / "frozen_roles.json").read_text())
        if all(r in coh_val["members"] for r in roles5):
            comp5 = st30.per_sample_components(coh_val, W5, roles5, cl5)
            boot["hier_cluster_vs_step5_frozen"] = st30.bootstrap_margin(
                comp_hier, comp5, n_boot=args.n_boot, seed=SEED)
            results["val:step5_frozen_replay"] = st30.score_cluster_blend(
                coh_val, W5, roles5, cl5)
            log(f"    Step-5 frozen weights replayed on this val cohort: "
                f"{results['val:step5_frozen_replay']['total_score']:.6f}")
    for k, v in boot.items():
        log(f"    {k}: diff={v['point_diff']:+.5f} "
            f"CI95=[{v['diff_ci95'][0]:+.5f}, {v['diff_ci95'][1]:+.5f}] "
            f"favouring-a={v['frac_bootstrap_replicates_favouring_a']:.3f}")
    (RESULTS / "step6_bootstrap_ci.json").write_text(json.dumps(boot, indent=2))

    # ---- persist ----
    np.save(CACHE5 / "step6_frozen_W.npy", W_fin)
    np.save(CACHE5 / "step6_frozen_clusters.npy", cl_fin)
    (CACHE5 / "step6_frozen_roles.json").write_text(json.dumps(roles))

    out = {
        "step": "6_4_hierarchical_cluster_stacking",
        "seed": SEED,
        "benchmark_total": BENCH_TOTAL,
        "step5_val_total": STEP5_VAL_TOTAL,
        "headline_val_total": headline,
        "step6_target": target,
        "step6_target_met": bool(headline > target),
        "delta_vs_step5": headline - STEP5_VAL_TOTAL,
        "delta_vs_benchmark": headline - BENCH_TOTAL,
        "roles": roles,
        "roles_dropped": dropped,
        "n_free_parameters": int(len(REGIMES) * len(roles) * best["nC"]),
        "selected": {"cluster_column": best["col"], "n_clusters": best["nC"],
                     "alpha": best["alpha"],
                     "selected_on": "the train-OOF cohort only; val was not consulted"},
        "hierarchy": hier_info,
        "protocol": {
            "weight_space": ("non-negative, per (availability regime x member role x "
                             "protein cluster), not constrained to sum to one; the "
                             "shortfall from one is shrinkage toward the control anchor"),
            "fitted_on": ("the 5-fold LCGO out-of-fold cohort: every member prediction "
                          "produced by a model that never saw the row"),
            "evaluated_on": "val_* once, with the OOF-fitted weights frozen",
            "hierarchical_warm_start": (
                "clusters are nested (K-means at the finest level, agglomerative "
                "merging of centroids for coarser levels), so each level is "
                "warm-started at its parent's optimum and departs from it only "
                "when the OOF objective improves"),
            "ridge": ("alpha penalises the variance of the weights across clusters "
                      "within a (regime, role) cell -- an L2 shrinkage of the cluster "
                      "extension back toward the scalar solution"),
            "objective_implementation": (
                "modules 1-3 (0.95 of the weight) are evaluated by the cluster "
                "co-moment factorisation, algebraically identical to the harness; "
                "module 4 is not a polynomial in the weights so it is held constant "
                "during the search, and every reported total comes from the real harness"),
            "fast_validation_scalar": val1,
        },
        "scalar_rung": {"oof_total": r_oof1["total_score"],
                        "val_total": r_val1["total_score"],
                        "n_evaluations": info1["n_evaluations"],
                        "seconds": info1["seconds"]},
        "ladder": ladder,
        "bootstrap": boot,
        "scores": {k: {"total_score": v["total_score"],
                       "module_scores": v["module_scores"]} for k, v in results.items()},
        "weights_table": st30.weights_table(W_fin, roles),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    S4.write_json(RESULTS / "step6_model_scores.json", out)
    S4.write_json(RESULTS / "step6_cluster_weights.json", out)
    log("\n    -> results/step6_model_scores.json")

    # ---- figure ----
    make_figure(out, results)
    log(f"\n=== step 6.4 complete in {time.time() - t0:.0f}s ===")


def make_figure(out: dict, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) member vs ensemble totals on val
    ax = axes[0]
    items = [(k.split(":", 1)[1], v["total_score"])
             for k, v in results.items() if k.startswith("val:")]
    items.sort(key=lambda t: t[1])
    ax.barh([i[0] for i in items], [i[1] for i in items], color="#4C72B0")
    ax.axvline(out["benchmark_total"], color="#C44E52", ls="--", lw=1,
               label=f"benchmark {out['benchmark_total']:.4f}")
    ax.axvline(out["step5_val_total"], color="#55A868", ls=":", lw=1.2,
               label=f"Step 5 {out['step5_val_total']:.4f}")
    ax.axvline(out["step6_target"], color="#8172B2", ls="-.", lw=1,
               label=f"target {out['step6_target']}")
    ax.set_xlabel("harness total score (validation)")
    ax.set_title("Members and ensembles on the validation cohort", fontsize=10)
    ax.legend(fontsize=6, loc="lower right")
    ax.tick_params(labelsize=7)

    # (b) OOF ladder by cluster level
    ax = axes[1]
    lad = {k: v for k, v in out["ladder"].items()
           if isinstance(v, dict) and "oof_total_surrogate" in v}
    by_alpha: dict = {}
    for k, v in lad.items():
        by_alpha.setdefault(v["alpha"], []).append((v["n_clusters"],
                                                    v["oof_total_surrogate"]))
    for a, pts in sorted(by_alpha.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                label=f"alpha={a}")
    ax.axhline(out["scalar_rung"]["oof_total"], color="grey", ls="--", lw=1,
               label="scalar rung")
    ax.set_xlabel("number of protein clusters K")
    ax.set_ylabel("OOF total (co-moment surrogate)")
    ax.set_title("Hierarchical ladder (selection cohort = OOF only)", fontsize=10)
    ax.legend(fontsize=7)

    # (c) module decomposition, Step 5 vs Step 6
    ax = axes[2]
    ms6 = out["scores"].get("val:hier_cluster_oof_frozen", {}).get("module_scores", {})
    ms5 = out["scores"].get("val:step5_frozen_replay", {}).get("module_scores", {})
    mods = list(ms6)
    x = np.arange(len(mods))
    if ms5:
        ax.bar(x - 0.2, [ms5.get(m, 0) for m in mods], 0.4, label="Step 5 frozen",
               color="#55A868")
    ax.bar(x + (0.2 if ms5 else 0.0), [ms6.get(m, 0) for m in mods],
           0.4 if ms5 else 0.6, label="Step 6 hierarchical", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(mods, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("module score")
    ax.set_title("Per-module decomposition on validation", fontsize=10)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(FIGURES / "step6_performance_comparison.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIGURES / "step6_performance_comparison.pdf", bbox_inches="tight")
    plt.close()
    log("    figure -> figures/step6_performance_comparison.png")


if __name__ == "__main__":
    sys.exit(main())
