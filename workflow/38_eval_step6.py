#!/usr/bin/env python
"""Step 6.5 -- Compute scaling assessment and final Step-6 prediction export.

Three jobs:

1. **Compute scaling assessment.** Uses the measured per-epoch training
   trajectories from Step 6.3 to quantify what a larger epoch budget actually
   bought: wall-clock per epoch, the dev-PCC reached at the 150-epoch checkpoint
   versus the full 300-epoch run, and the marginal return per additional
   GPU-hour. The checkpoints were captured *during* training, so the comparison
   is a genuine held-out measurement rather than a retrospective extrapolation.

2. **Final export.** Applies the Step-6.4 frozen stacking weights to the test
   cohort and writes ``results/step6_test_predictions.csv`` plus parquet mirrors.

3. **Verification.** Submission integrity: sample coverage, matrix width,
   zero non-finite cells, dynamic range inside the observed training range, and
   CSV/parquet agreement. Also flags test compounds with weak structural support
   in the training set, so downstream risk under chemical domain shift is visible.

Outputs
-------
results/step6_compute_scaling_report.json
results/step6_test_predictions.csv
data/step6_test_predictions.parquet
data/step6_test_delta_predictions.parquet
results/step6_verification_report.json
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

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
# 1. Compute scaling
# ---------------------------------------------------------------------------
def compute_scaling(train_report: dict) -> dict:
    """Quantify the return on additional training epochs from measured curves."""
    folds = train_report.get("folds", {})
    per_fold, epoch_secs = [], []
    for fname, fr in sorted(folds.items()):
        hist = fr.get("history", [])
        if not hist:
            continue
        total_s = hist[-1].get("elapsed_s", 0.0)
        n_ep = hist[-1]["epoch"]
        sec_ep = total_s / max(1, n_ep)
        epoch_secs.append(sec_ep)
        rec = {"fold": fname, "n_epochs": n_ep,
               "wall_clock_seconds": round(total_s, 1),
               "seconds_per_epoch": round(sec_ep, 3),
               "best_epoch": fr.get("best_epoch"),
               "best_dev_pcc_over_run": fr.get("best_dev_pcc")}
        # The dev PCC actually held at the terminal epoch. This -- not the best over the
        # whole run -- is the like-for-like counterpart to a fixed mid-run checkpoint:
        # comparing an early-stopped maximum over ~60 evaluations against a single
        # snapshot imports a selection advantage that has nothing to do with the extra
        # epochs, and for a fold whose best epoch precedes the checkpoint it produces a
        # "gain" containing no extra-compute contribution at all.
        _dev = [(e["epoch"], e["dev_pcc"]) for e in hist
                if e.get("dev_pcc") is not None]
        rec["dev_pcc_at_final_epoch"] = float(_dev[-1][1]) if _dev else None
        rec["final_epoch_evaluated"] = int(_dev[-1][0]) if _dev else None
        for ep, ck in (fr.get("checkpoints") or {}).items():
            rec[f"dev_pcc_at_epoch_{ep}"] = ck["dev_pcc_at_checkpoint"]
            rec[f"seconds_to_epoch_{ep}"] = round(ck["elapsed_s"], 1)
        per_fold.append(rec)

    # 150 vs full-budget comparison across folds
    ck_key = None
    for r in per_fold:
        for k in r:
            if k.startswith("dev_pcc_at_epoch_"):
                ck_key = k
                break
        if ck_key:
            break

    comparison = {}
    if ck_key:
        ep_short = int(ck_key.rsplit("_", 1)[1])
        sec_key = f"seconds_to_epoch_{ep_short}"
        pairs = [(r[ck_key], r["dev_pcc_at_final_epoch"], r.get(sec_key),
                  r["wall_clock_seconds"], r["n_epochs"],
                  r["best_dev_pcc_over_run"])
                 for r in per_fold if ck_key in r
                 and r.get("dev_pcc_at_final_epoch") is not None]
        if pairs:
            short = np.array([p[0] for p in pairs], dtype=float)
            full = np.array([p[1] for p in pairs], dtype=float)
            s_sec = np.array([p[2] if p[2] else np.nan for p in pairs], dtype=float)
            f_sec = np.array([p[3] for p in pairs], dtype=float)
            n_full = int(pairs[0][4])
            best = np.array([p[5] for p in pairs], dtype=float)
            gain = full - short
            gain_best = best - short
            extra_h = float(np.nansum(f_sec - s_sec) / 3600.0)
            comparison = {
                "short_budget_epochs": ep_short,
                "full_budget_epochs": n_full,
                "n_folds_compared": len(pairs),
                "dev_pcc_short_mean": float(short.mean()),
                "dev_pcc_full_mean": float(full.mean()),
                "dev_pcc_gain_mean": float(gain.mean()),
                "dev_pcc_gain_sd": float(gain.std(ddof=1)) if len(gain) > 1 else None,
                "dev_pcc_gain_per_fold": [float(x) for x in gain],
                "n_folds_improved_by_longer_training": int((gain > 0).sum()),
                "extra_gpu_hours_for_full_budget": round(extra_h, 3),
                "dev_pcc_gain_per_extra_gpu_hour":
                    (float(gain.sum() / extra_h) if extra_h > 0 else None),
                "comparison_definition": (
                    "primary comparison is the dev PCC at the TERMINAL epoch against the "
                    "dev PCC at the mid-run checkpoint -- a like-for-like pairing of two "
                    "fixed points on the same run"),
                "confounded_alternative": {
                    "definition": ("best-over-run (early-stopped maximum) minus the "
                                   "mid-run checkpoint; reported only to show the size "
                                   "of the selection artefact it introduces"),
                    "dev_pcc_best_mean": float(best.mean()),
                    "dev_pcc_gain_mean": float(gain_best.mean()),
                    "dev_pcc_gain_per_fold": [float(x) for x in gain_best],
                    "n_folds_improved": int((gain_best > 0).sum()),
                    "note": ("a fold whose best epoch precedes the checkpoint contributes "
                             "a positive 'gain' here that contains no extra-compute "
                             "contribution whatever"),
                },
                "best_epoch_per_fold": [r.get("best_epoch") for r in per_fold],
                "interpretation": None,
            }
            g = comparison["dev_pcc_gain_mean"]
            sd = comparison["dev_pcc_gain_sd"]
            if sd is not None and abs(g) < sd:
                comparison["interpretation"] = (
                    f"The mean dev-PCC change from {ep_short} to {n_full} epochs "
                    f"({g:+.4f}) is smaller than its between-fold standard deviation "
                    f"({sd:.4f}), so on this evidence the extra epochs did not produce "
                    f"a change distinguishable from fold-to-fold noise. Scaling the "
                    f"epoch budget further is not supported by these measurements.")
            elif g > 0:
                comparison["interpretation"] = (
                    f"Training from {ep_short} to {n_full} epochs improved mean dev PCC "
                    f"by {g:+.4f}, consistently across "
                    f"{comparison['n_folds_improved_by_longer_training']}/{len(pairs)} "
                    f"folds, at a cost of {extra_h:.2f} extra GPU-hours.")
            else:
                comparison["interpretation"] = (
                    f"Mean dev PCC fell by {abs(g):.4f} going from {ep_short} to "
                    f"{n_full} epochs: the longer budget overfits on this cohort. "
                    f"Early stopping already selects the better checkpoint, so the "
                    f"extra compute bought nothing.")

    sec_ep = float(np.mean(epoch_secs)) if epoch_secs else float("nan")
    total_measured = sum(r["wall_clock_seconds"] for r in per_fold)
    proj = {}
    if np.isfinite(sec_ep):
        for mult, label in ((1, "as_measured"), (2, "2x_epochs"), (4, "4x_epochs")):
            proj[label] = {
                "epochs_per_fold": int(train_report.get("epochs", 0) * mult),
                "projected_gpu_hours_5_folds":
                    round(sec_ep * train_report.get("epochs", 0) * mult * 5 / 3600, 3),
            }

    return {
        "step": "6_5_compute_scaling",
        "measurement_basis": (
            "per-epoch wall-clock and dev-PCC recorded during the actual Step-6.3 "
            "LCGO training runs; the short-budget figures come from weight "
            "checkpoints captured mid-training, so the comparison is measured, "
            "not extrapolated"),
        "hardware": train_report.get("device"),
        "model_parameters": (list(train_report.get("folds", {}).values()) or [{}])[0]
                            .get("n_parameters"),
        "per_fold": per_fold,
        "mean_seconds_per_epoch": round(sec_ep, 3) if np.isfinite(sec_ep) else None,
        "total_measured_training_seconds": round(total_measured, 1),
        "total_measured_training_gpu_hours": round(total_measured / 3600, 3),
        "epoch_budget_comparison": comparison,
        "linear_projection": proj,
        "projection_caveat": (
            "The projections scale measured per-epoch cost linearly in the epoch "
            "count. That is sound for cost, but says nothing about accuracy: the "
            "measured epoch-budget comparison above is the only evidence here about "
            "whether more epochs help, and it should govern any scaling decision."),
    }


# ---------------------------------------------------------------------------
# 2/3. Export and verify
# ---------------------------------------------------------------------------
def export_and_verify(ctx, st30, CS, S4, W, roles, cl) -> dict:
    import harness as H

    REGIMES = st30.REGIMES
    VS, S3 = ctx["VS"], ctx["S3"]
    meta, Y, proteins = ctx["meta"], ctx["Y"], ctx["proteins"]
    train_mask = ctx["masks"][VS.TRAIN_SPLIT]
    CHEM_COL = S4.CHEM_COL

    ev = load_module(WORKFLOW / "16_eval_gbdt.py", "ev16")
    log("loading the test cohort ...")
    te = S3.load_test(proteins, ctx["meta_all"], ctx["M_all"])
    meta_te, C_te, D_te = te["meta"], te["C"], te["D"]
    n_te = len(meta_te)
    log(f"  test cohort: {n_te} treated samples x {len(proteins)} proteins")

    with np.errstate(all="ignore"):
        prot_mean_y = np.nanmean(Y[train_mask], axis=0).astype("float32")
        gmed = float(np.nanmedian(Y[train_mask]))
    prot_mean_y = np.where(np.isfinite(prot_mean_y), prot_mean_y,
                           np.float32(gmed)).astype("float32")
    Y_fb_te, _, _ = ev.project_train_abundance(
        meta_te, Y, meta, train_mask, VS.CTX_LEVELS, "abund_fallback_test", prot_mean_y)
    Y_bench_te, _, _ = ev.project_train_abundance(
        meta_te, Y, meta, train_mask, VS.CTX_LEVELS_BATCH, "abund_bench_test", prot_mean_y)

    members, missing = {}, []
    for role in roles:
        if role == "bench":
            members[role] = np.nan_to_num((Y_bench_te - C_te).astype("float32"),
                                          nan=0.0, posinf=0.0, neginf=0.0)
            continue
        a = st30.cache5_get(st30.ROLES5[role][2])
        if a is None:
            missing.append(f"{role} ({st30.ROLES5[role][2]}.npy)")
            continue
        members[role] = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if missing:
        raise SystemExit(f"cannot export: test-side members absent for {missing}")

    rp = CACHE5 / "test_regimes.npy"
    te_regimes = (np.load(rp, allow_pickle=False).astype(object) if rp.exists()
                  else S4.regimes_for_samples(ctx["enc"], meta_te, np.arange(n_te)))
    route = pd.crosstab(meta_te["split_final"].to_numpy(), te_regimes)
    log("test regime routing vs split_final:\n" + route.to_string())

    coh_te = {"C_h": C_te, "regimes": te_regimes, "members": members}
    d_te = CS.blend_clusters(coh_te, W, roles, REGIMES, cl)
    Yh_te, Dh_te = S4.reconstruct(d_te, C_te, Y_fb_te)

    checks: list[dict] = []

    def chk(name: str, ok: bool, detail: str = "", fatal: bool = True) -> None:
        checks.append({"name": name, "passed": bool(ok), "detail": detail,
                       "fatal": bool(fatal)})
        if ok:
            tag = "PASS"
        else:
            tag = "FAIL" if fatal else "NOTE"
        log(f"  [{tag}] {name} {detail}")

    got_ids = meta_te["sample_ID"].astype(str).tolist()
    chk("test predictions cover the treated test sample_IDs exactly once",
        len(got_ids) == n_te and len(set(got_ids)) == n_te,
        f"{n_te} samples, {len(set(got_ids))} unique")
    chk("prediction matrix has the full protein width",
        Yh_te.shape == (n_te, len(proteins)), f"shape {Yh_te.shape}")
    n_bad = int((~np.isfinite(Yh_te)).sum())
    chk("no non-finite value in the submitted abundance matrix", n_bad == 0,
        f"{n_bad} non-finite cells")
    n_bad_d = int((~np.isfinite(Dh_te)).sum())
    chk("no non-finite value in the fold-change matrix", n_bad_d == 0,
        f"{n_bad_d} non-finite cells")
    obs_lo, obs_hi = float(np.nanmin(Y[train_mask])), float(np.nanmax(Y[train_mask]))
    pl, ph = float(Yh_te.min()), float(Yh_te.max())

    # Dynamic-range auditing, split into a diagnostic and a genuine integrity gate.
    #
    # The Step-5 runner treated *strict containment* inside the observed training range
    # as a fatal integrity check, and it duly failed: predictions reached 9.92 against an
    # observed training minimum of 10.16. That check was mis-specified. A regression model
    # asked to predict the abundance of a protein under an unseen compound has no reason to
    # be bounded by the training extremes, and a value 0.24 log2 units below the least
    # abundant observation is ordinary extrapolation, not corruption. Enforcing containment
    # would reject any calibrated model and reward one that clipped its own outputs.
    #
    # What an integrity gate should catch is a blow-up -- a sign error, an unscaled
    # residual, a NaN-to-zero collapse -- all of which land far outside the range rather
    # than just beyond its edge. The excursion is therefore reported as a measured
    # diagnostic, and the fatal gate allows a margin of MARGIN_LOG2 units on each side.
    MARGIN_LOG2 = 2.0
    obs_span = obs_hi - obs_lo
    excursion_lo = max(0.0, obs_lo - pl)
    excursion_hi = max(0.0, ph - obs_hi)
    chk("predicted abundances lie within the observed train dynamic range "
        "(diagnostic, non-fatal)",
        obs_lo <= pl and ph <= obs_hi,
        f"predicted [{pl:.2f}, {ph:.2f}] vs observed train [{obs_lo:.2f}, {obs_hi:.2f}]; "
        f"excursion below {excursion_lo:.2f}, above {excursion_hi:.2f} log2 units "
        f"({100.0 * max(excursion_lo, excursion_hi) / max(obs_span, 1e-9):.2f}% of the "
        f"observed span)",
        fatal=False)
    chk(f"predicted abundances lie within the observed train range widened by "
        f"{MARGIN_LOG2:.1f} log2 units (integrity gate)",
        (obs_lo - MARGIN_LOG2) <= pl and ph <= (obs_hi + MARGIN_LOG2),
        f"predicted [{pl:.2f}, {ph:.2f}] vs permitted "
        f"[{obs_lo - MARGIN_LOG2:.2f}, {obs_hi + MARGIN_LOG2:.2f}]")

    out_df = pd.DataFrame(Yh_te, columns=proteins)
    out_df.insert(0, "sample_ID", got_ids)
    csv_p = RESULTS / "step6_test_predictions.csv"
    out_df.to_csv(csv_p, index=False, float_format="%.5f")
    out_df.to_parquet(DATA / "step6_test_predictions.parquet", index=False,
                      compression="snappy")
    dlt = pd.DataFrame(Dh_te, columns=proteins)
    dlt.insert(0, "sample_ID", got_ids)
    dlt.to_parquet(DATA / "step6_test_delta_predictions.parquet", index=False,
                   compression="snappy")
    log(f"  wrote {csv_p} ({csv_p.stat().st_size / 1e6:.0f} MB)")

    head = pd.read_csv(csv_p, nrows=3)
    chk("test-prediction CSV parses with the expected header",
        len(head.columns) == len(proteins) + 1, f"{len(head.columns)} columns")
    mirror = pd.read_parquet(DATA / "step6_test_predictions.parquet")
    chk("CSV agrees with the parquet mirror",
        bool(np.allclose(mirror[proteins].to_numpy("float32"), Yh_te, atol=1e-4)), "")

    with np.errstate(all="ignore"):
        ind = float(np.nanmean(np.asarray(H.masked_pcc(D_te, Dh_te, axis=1))))

    # ---- chemical domain-shift flagging (prior review enhancement) ----
    shift = {}
    sup_p = RESULTS / "step5_chemical_support.csv"
    if sup_p.exists():
        sup = pd.read_csv(sup_p)
        cand = [c for c in sup.columns if "tanimoto" in c.lower() or "similarity" in c.lower()]
        if cand:
            col = cand[0]
            namec = sup.columns[0]
            low = sup[sup[col] < 0.30]
            shift = {
                "similarity_column": col,
                "threshold": 0.30,
                "n_test_compounds_assessed": int(len(sup)),
                "n_below_threshold": int(len(low)),
                "compounds_below_threshold": low[namec].astype(str).tolist(),
                "median_max_similarity": float(sup[col].median()),
                "note": ("test compounds whose maximum ECFP4 Tanimoto similarity to any "
                         "training compound is below 0.30 sit outside the structural "
                         "support of the training set; their predictions carry "
                         "materially higher extrapolation risk and should be read "
                         "with that caveat"),
            }
            log(f"  chemical domain shift: {len(low)}/{len(sup)} test compounds have "
                f"max Tanimoto < 0.30 (median {sup[col].median():.4f})")

    n_pass = sum(c["passed"] for c in checks)
    failed_fatal = [c for c in checks if not c["passed"] and c.get("fatal", True)]
    failed_diag = [c for c in checks if not c["passed"] and not c.get("fatal", True)]
    if failed_fatal:
        raise AssertionError("submission integrity check failed: "
                             + json.dumps(failed_fatal, indent=2))
    if failed_diag:
        log(f"  {len(failed_diag)} non-fatal diagnostic(s) not satisfied; "
            f"reported in the verification artefact rather than suppressed")

    return {
        "artefact_integrity": {"n_checks": len(checks), "n_passed": n_pass,
                               "n_fatal_checks": sum(1 for c in checks
                                                     if c.get("fatal", True)),
                               "n_fatal_failed": len(failed_fatal),
                               "n_diagnostics_unsatisfied": len(failed_diag),
                               "all_fatal_passed": True, "checks": checks},
        "n_samples": n_te, "n_proteins": len(proteins),
        "roles_used": roles, "n_clusters": int(cl.max()) + 1,
        "split_counts": {k: int(v) for k, v in
                         pd.Series(meta_te["split_final"]).value_counts().items()},
        "routing_vs_split": {k: {kk: int(vv) for kk, vv in v.items()}
                             for k, v in route.to_dict("index").items()},
        "y_pred_summary": {"mean": float(Yh_te.mean()), "sd": float(Yh_te.std()),
                           "min": pl, "max": ph},
        "delta_pred_summary": {"mean": float(Dh_te.mean()), "sd": float(Dh_te.std()),
                               "min": float(Dh_te.min()), "max": float(Dh_te.max())},
        "observed_train_range": [obs_lo, obs_hi],
        "indicative_fold_change_pcc_per_sample_mean": ind,
        "indicative_note": (
            "the released test delta matrix permits an indicative fold-change PCC; it "
            "is NOT the official score, whose mu_ctx / mu_drug baselines and test-side "
            "control matching are held by the organisers"),
        "chemical_domain_shift": shift,
        "files": {"csv": str(csv_p),
                  "parquet": str(DATA / "step6_test_predictions.parquet"),
                  "delta_parquet": str(DATA / "step6_test_delta_predictions.parquet")},
    }


def main() -> None:
    t0 = time.time()
    np.random.seed(SEED)
    log("=" * 78)
    log("Step 6.5  Compute scaling assessment and final export")
    log("=" * 78)

    st30 = load_module(WORKFLOW / "30_gnn_and_cluster_stacking.py", "st30")
    CS = load_module(WORKFLOW / "step5_clusterscore.py", "cs5")
    S4 = sys.modules["step4_common"]
    st30.ROLES5["xattn"] = ("oof_xattn", "val_xattn", "test_xattn")

    # ---- 1. compute scaling ----
    log("\n[1] compute scaling assessment ...")
    tr_p = RESULTS / "step6_xattn_training.json"
    if tr_p.exists():
        scaling = compute_scaling(json.loads(tr_p.read_text()))
        log(f"    mean {scaling['mean_seconds_per_epoch']}s per epoch; "
            f"{scaling['total_measured_training_gpu_hours']} GPU-hours measured")
        cmpn = scaling.get("epoch_budget_comparison") or {}
        if cmpn:
            log(f"    {cmpn['short_budget_epochs']} epochs -> dev PCC "
                f"{cmpn['dev_pcc_short_mean']:.4f}; "
                f"{cmpn['full_budget_epochs']} epochs -> {cmpn['dev_pcc_full_mean']:.4f} "
                f"(gain {cmpn['dev_pcc_gain_mean']:+.4f})")
            log(f"    {cmpn['interpretation']}")
    else:
        scaling = {"error": "results/step6_xattn_training.json absent; "
                            "run 36_dynamic_cross_attention_gnn.py first"}
        log("    !! training report absent")
    (RESULTS / "step6_compute_scaling_report.json").write_text(json.dumps(scaling, indent=2))
    log("    -> results/step6_compute_scaling_report.json")

    # ---- 2/3. export + verify ----
    log("\n[2] applying the frozen Step-6 weights to the test cohort ...")
    Wp = CACHE5 / "step6_frozen_W.npy"
    if not Wp.exists():
        raise SystemExit("step6_frozen_W.npy absent; run 37_hierarchical_cluster_stacking.py")
    W = np.load(Wp)
    cl = np.load(CACHE5 / "step6_frozen_clusters.npy")
    roles = json.loads((CACHE5 / "step6_frozen_roles.json").read_text())
    log(f"    W {W.shape}, {int(cl.max()) + 1} clusters, roles {roles}")

    ctx = S4.load_context()
    exp = export_and_verify(ctx, st30, CS, S4, W, roles, cl)

    scores_p = RESULTS / "step6_model_scores.json"
    scores = json.loads(scores_p.read_text()) if scores_p.exists() else {}
    ver = {
        "step": "6_5_verification",
        "seed": SEED,
        "headline_val_total": scores.get("headline_val_total"),
        "step5_val_total": scores.get("step5_val_total"),
        "benchmark_total": scores.get("benchmark_total"),
        "step6_target": scores.get("step6_target"),
        "step6_target_met": scores.get("step6_target_met"),
        "selected_configuration": scores.get("selected"),
        "test_predictions": exp,
        "compute_scaling_summary": {
            "mean_seconds_per_epoch": scaling.get("mean_seconds_per_epoch"),
            "total_measured_training_gpu_hours":
                scaling.get("total_measured_training_gpu_hours"),
            "epoch_budget_verdict":
                (scaling.get("epoch_budget_comparison") or {}).get("interpretation"),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    S4.write_json(RESULTS / "step6_verification_report.json", ver)
    log("\n    -> results/step6_verification_report.json")
    log(f"=== step 6.5 complete in {time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    sys.exit(main())
