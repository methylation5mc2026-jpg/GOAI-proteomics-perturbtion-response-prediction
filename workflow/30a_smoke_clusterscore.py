"""Step 5.4-smoke -- prove the cluster co-moment scorer is exact before using it.

Runs two independent checks on the validation cohort, using the Step-4 cached
members so it needs nothing from the still-running LCGO stage:

1. **Harness agreement.** :meth:`ClusterFastScorer.validate` compares modules 1-3
   against ``harness.compute_competition_score`` on five cluster-weight tensors,
   including a deliberately adversarial one that alternates 0 and 1.2 between
   neighbouring clusters. A scatter-accumulation bug would cancel out under any
   cluster-constant tensor, so the constant case alone would not catch it.
2. **Reduction to Step 4.** With ``n_clusters = 1`` the cluster scorer must
   reproduce :class:`step4_fastscore.FastScorer` to floating-point noise, and a
   cluster-constant tensor at ``n_clusters = 8`` must give the same module scores
   as the equivalent scalar weights. That is what makes the scalar rung of the
   ablation ladder a genuine control rather than a differently-implemented number.

Writes results/step5_clusterscore_smoke.json. A failure raises.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_common as S4  # noqa: E402
import step4_fastscore as FS  # noqa: E402
import step5_clusterscore as CS  # noqa: E402

DATA, RESULTS = S4.DATA, S4.RESULTS
SEED, log = S4.SEED, S4.log
REGIMES = ("full", "chem_novel", "strain_novel", "both_novel")

#: Step-4 member roles, used here purely as a fixture: this script tests the
#: scorer's algebra, not any Step-5 model.
ROLES4 = {
    "gbdt_tab": "val_lgb_delta",
    "gbdt_mol": "val_lgb_rdkit_delta",
    "bench": "val_bench_delta",
    "dl": "dl_delta_val",
}


def main() -> None:
    np.random.seed(SEED)
    log("=== Step 5.4 smoke: cluster co-moment scorer exactness ===")

    ctx = S4.load_context()
    meta, Y, D = ctx["meta"], ctx["Y"], ctx["D"]
    mask = ctx["masks"]["all_val"]
    idx = np.flatnonzero(mask)

    members = {}
    for role, name in ROLES4.items():
        a = S4.cache_get(name)
        if a is not None:
            members[role] = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    roles = sorted(members)
    log(f"fixture roles on val: {roles}")

    y_fb = S4.cache_get("val_y_fallback")
    if y_fb is None:
        y_fb = ctx["y_fallback"][mask]
    rp = S4.cache_path("val_regimes")
    regimes = (np.load(rp, allow_pickle=False).astype(object) if rp.exists()
               else S4.regimes_for_samples(ctx["enc"], meta, idx))

    coh = {
        "idx": idx, "meta_eval": meta.loc[mask].reset_index(drop=True),
        "Y": Y[idx], "D": D[idx], "C_h": ctx["C_harness"][idx], "y_fb": y_fb,
        "mu_ctx": ctx["mu_ctx"][mask], "mu_drug": ctx["mu_drug"][mask],
        "members": members, "regimes": regimes,
    }
    log(f"val cohort: {len(idx)} samples x {Y.shape[1]} proteins, "
        f"regimes {dict(pd.Series(regimes).value_counts())}")

    clus = pd.read_parquet(DATA / "step5_protein_clusters.parquet")
    cl8 = clus["k8"].to_numpy(np.int64)
    nC = int(cl8.max()) + 1
    ones = np.zeros(Y.shape[1], dtype=np.int64)
    log(f"cluster index k8: {nC} clusters, sizes {np.bincount(cl8).tolist()}")

    def exact(W, cl):
        d = CS.blend_clusters(coh, W, roles, REGIMES, cl)
        yh, dh = S4.reconstruct(d, coh["C_h"], coh["y_fb"])
        return S4.score(coh["Y"], yh, coh["D"], dh, coh["meta_eval"],
                        coh["mu_ctx"], coh["mu_drug"])

    out: dict = {"step": "5_4_smoke", "seed": SEED, "roles": roles,
                 "n_val_samples": int(len(idx)), "n_clusters": nC,
                 "cluster_sizes": np.bincount(cl8).tolist()}

    # ---- check 1: harness agreement at n_clusters = 8 --------------------
    log(f"\n[1] building the cluster scorer at n_clusters={nC} ...")
    t0 = time.time()
    fast8 = CS.ClusterFastScorer(coh, roles, REGIMES, cl8, nC)
    build8 = time.time() - t0
    log(f"    built in {build8:.1f}s")
    v8 = fast8.validate(lambda W: exact(W, cl8), n_trials=5, seed=SEED)
    out["check1_harness_agreement_k8"] = {k: v for k, v in v8.items() if k != "rows"}
    out["check1_build_seconds"] = round(build8, 2)

    # ---- check 2a: n_clusters = 1 reproduces the Step-4 scorer ------------
    log("\n[2a] building the cluster scorer at n_clusters=1 and the Step-4 scorer ...")
    t0 = time.time()
    fast1 = CS.ClusterFastScorer(coh, roles, REGIMES, ones, 1)
    build1 = time.time() - t0
    fast4 = FS.FastScorer(coh, roles, REGIMES, verbose=False)
    rng = np.random.default_rng(SEED)
    worst = 0.0
    rows = []
    for t in range(4):
        Wm = (np.full((len(REGIMES), len(roles)), 0.4) if t == 0
              else rng.uniform(0.0, 1.1, size=(len(REGIMES), len(roles))))
        a = fast1.module_scores(Wm[:, :, None])
        b = fast4.module_scores(Wm)
        for m in a:
            d = abs(a[m] - b[m])
            worst = max(worst, d)
            rows.append({"trial": t, "module": m, "cluster_nC1": a[m],
                         "step4": b[m], "abs_diff": d})
        log(f"    trial {t}: max |cluster(nC=1) - step4| = "
            f"{max(r['abs_diff'] for r in rows if r['trial'] == t):.3e}")
    if worst > 1e-9:
        raise AssertionError(
            f"n_clusters=1 does not reproduce the Step-4 scorer (max dev {worst:.3e}); "
            f"the scalar rung of the ablation ladder would not be a valid control. "
            f"Worst: {sorted(rows, key=lambda r: -r['abs_diff'])[:3]}"
        )
    log(f"    PASS: n_clusters=1 reproduces step4_fastscore (max dev {worst:.3e})")
    out["check2a_reduces_to_step4_scorer"] = {
        "max_abs_deviation": float(worst), "tol": 1e-9, "n_trials": 4,
        "build_seconds": round(build1, 2),
    }

    # ---- check 2b: a cluster-constant tensor equals the scalar weights ----
    log("\n[2b] cluster-constant tensor at nC=8 vs the equivalent scalar weights ...")
    worst2 = 0.0
    rows2 = []
    for t in range(3):
        Wm = rng.uniform(0.0, 0.9, size=(len(REGIMES), len(roles)))
        a = fast8.module_scores(np.repeat(Wm[:, :, None], nC, axis=2))
        b = fast1.module_scores(Wm[:, :, None])
        for m in a:
            d = abs(a[m] - b[m])
            worst2 = max(worst2, d)
            rows2.append({"trial": t, "module": m, "abs_diff": d})
        log(f"    trial {t}: max |k8 constant - nC=1| = "
            f"{max(r['abs_diff'] for r in rows2 if r['trial'] == t):.3e}")
    if worst2 > 1e-9:
        raise AssertionError(
            f"a cluster-constant weight tensor disagrees with the scalar scorer by "
            f"{worst2:.3e}; the cluster partition or the accumulation is wrong"
        )
    log(f"    PASS: cluster-constant == scalar (max dev {worst2:.3e})")
    out["check2b_cluster_constant_equals_scalar"] = {
        "max_abs_deviation": float(worst2), "tol": 1e-9, "n_trials": 3,
    }

    # ---- check 2c: incremental block updates stay exact -------------------
    # The scorer updates only the (regime, cluster) blocks a candidate touches.
    # That is what makes the search feasible, but it also means round-off could
    # in principle accumulate over a long coordinate-ascent run. So: walk 3,000
    # single-coordinate perturbations through the incremental path, then compare
    # against a scorer that has never seen any of them.
    log("\n[2c] 3,000 single-coordinate updates vs a cold scorer ...")
    Wc = np.full((len(REGIMES), len(roles), nC), 0.3)
    _ = fast8.module_scores(Wc)
    for step in range(3000):
        ri = int(rng.integers(len(REGIMES)))
        ki = int(rng.integers(len(roles)))
        ci = int(rng.integers(nC))
        Wc[ri, ki, ci] = float(rng.uniform(0.0, 1.0))
        _ = fast8.total(Wc, m4=0.0)
        if (step + 1) % 1000 == 0:
            log(f"    {step + 1}/3000 incremental updates")
    warm = fast8.module_scores(Wc)
    cold = CS.ClusterFastScorer(coh, roles, REGIMES, cl8, nC,
                               verbose=False).module_scores(Wc)
    worst3 = max(abs(warm[m] - cold[m]) for m in warm)
    log(f"    max |warm(incremental) - cold(full)| = {worst3:.3e}")
    if worst3 > 1e-9:
        raise AssertionError(
            f"incremental block updates drifted by {worst3:.3e} after 3,000 "
            f"perturbations; the fast path is not exact"
        )
    log(f"    PASS: incremental path is exact "
        f"({fast8.tasks['m1_p'].n_block_updates:,} block updates, "
        f"{fast8.tasks['m1_p'].n_full_recomputes} full recomputes on task m1_p)")
    out["check2c_incremental_updates_exact"] = {
        "n_perturbations": 3000,
        "max_abs_deviation_vs_cold_scorer": float(worst3),
        "tol": 1e-9,
        "block_updates_on_task_m1_p": int(fast8.tasks["m1_p"].n_block_updates),
        "full_recomputes_on_task_m1_p": int(fast8.tasks["m1_p"].n_full_recomputes),
    }

    # ---- timing: is the cluster search actually feasible? ----------------
    # Timed the way the optimiser actually calls it: one coordinate changed per
    # evaluation. Timing a repeated identical W would measure the no-op path.
    W = np.full((len(REGIMES), len(roles), nC), 0.3)
    _ = fast8.total(W, m4=0.0)
    t0 = time.time()
    n = 400
    for i in range(n):
        W[i % len(REGIMES), (i // 2) % len(roles), (i // 3) % nC] = 0.2 + 0.001 * i
        fast8.total(W, m4=0.0)
    per = (time.time() - t0) / n
    log(f"\n[3] {1000 * per:.2f} ms per cluster-weight evaluation "
        f"({len(REGIMES) * len(roles) * nC} free parameters)")
    out["check3_seconds_per_evaluation"] = per
    out["n_free_parameters"] = len(REGIMES) * len(roles) * nC
    out["exact_harness_seconds_per_evaluation_for_scale"] = None

    t0 = time.time()
    _ = exact(np.full((len(REGIMES), len(roles), nC), 0.3), cl8)
    ex = time.time() - t0
    out["exact_harness_seconds_per_evaluation_for_scale"] = round(ex, 2)
    log(f"    one exact harness pass costs {ex:.1f}s -> speed-up "
        f"{ex / max(per, 1e-9):.0f}x")

    out["all_checks_passed"] = True
    S4.write_json(RESULTS / "step5_clusterscore_smoke.json", out)
    log("=== smoke test PASSED ===")


if __name__ == "__main__":
    main()
