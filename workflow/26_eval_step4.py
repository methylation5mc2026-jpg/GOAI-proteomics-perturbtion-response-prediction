"""Step 4.5 -- test-set predictions, verification report and Step-4 figures.

The submitted quantity is the **abundance** matrix. The fold-change is derived
from it by the organisers, so this script keeps the two coupled
(``y = C + Delta``) rather than optimising them separately -- decoupling would
inflate the local score in a way that could not be reproduced at submission time.

Verification is split into two categories that are reported separately and must
not be conflated:

* **artefact integrity** -- shape, ordering, finiteness, CSV/parquet agreement,
  JSON validity. A failure here is a *defect*, so it raises.
* **success criteria** -- whether the score targets were met. A failure here is a
  *scientific result*, so it is recorded and reported, never raised.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_common as S4  # noqa: E402

DATA, RESULTS, FIGURES = S4.DATA, S4.RESULTS, S4.FIGURES
WORKFLOW, MODELS4, SEED, log = S4.WORKFLOW, S4.MODELS4, S4.SEED, S4.log

REGIMES = ("full", "chem_novel", "strain_novel", "both_novel")
BENCH_TOTAL = 0.445442994


def load_module(path: Path, name: str):
    """Import a module whose filename starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_specialists(kind: str) -> dict | None:
    """Load the four LightGBM regime specialists of one flavour."""
    import lightgbm as lgb

    got = {}
    for regime in REGIMES:
        p = (
            WORKFLOW / "models" / f"lgb_delta__{regime}.txt"
            if kind == "tab"
            else MODELS4 / f"lgb_rdkit_delta__{regime}.txt"
        )
        if not p.exists():
            return None
        got[regime] = (lgb.Booster(model_file=str(p)), "lgb")
    return got


# ---------------------------------------------------------------------------
def make_performance_figure(scores: dict) -> None:
    """Per-module and total comparison of the Step-4 candidates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    mods = list(scores["module_weights"])
    weights = scores["module_weights"]

    show = [
        ("val:stacked_oof_frozen", "Step 4 stacked (frozen OOF weights)", "#2E7D32"),
        ("val:bench", "group-mean bench member", "#C4442E"),
        ("val:gbdt_tab", "Step 3 tabular LightGBM", "#3B6EA8"),
        ("val:gbdt_mol", "Step 4 RDKit LightGBM", "#7B4EA8"),
        ("val:dl", "Step 4 MLP-ResNet", "#B8860B"),
        ("val:control_anchor", "control anchor (Delta = 0 null)", "#888888"),
    ]
    show = [s for s in show if s[0] in scores["module_scores"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3),
                             gridspec_kw={"width_ratios": [2.1, 1]})

    ax = axes[0]
    x = np.arange(len(mods))
    bw = 0.8 / max(len(show), 1)
    for i, (key, label, col) in enumerate(show):
        vals = [scores["module_scores"][key].get(m, np.nan) for m in mods]
        ax.bar(x + i * bw - 0.4 + bw / 2, vals, bw, label=label, color=col, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(w={weights[m]:.2f})" for m in mods], fontsize=6.5)
    ax.set_ylabel("module score")
    ax.set_title("Per-module score on the validation cohort")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.set_ylim(0, 1.0)

    ax = axes[1]
    tot = {k: scores["totals"][k] for k, _, _ in show if k in scores["totals"]}
    order = sorted(tot, key=lambda k: tot[k])
    labels = {k: lb for k, lb, _ in show}
    colors = {k: c for k, _, c in show}
    ax.barh(
        range(len(order)),
        [tot[k] for k in order],
        color=[colors[k] for k in order],
        lw=0,
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[k] for k in order], fontsize=6.5)
    ax.axvline(BENCH_TOTAL, color="k", ls="--", lw=1.0)
    ax.annotate(
        f"official benchmark\n{BENCH_TOTAL:.4f}",
        xy=(BENCH_TOTAL, len(order) - 0.55),
        xytext=(BENCH_TOTAL - 0.11, len(order) - 0.30),
        fontsize=6,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.6, color="k"),
    )
    ax.set_xlabel("weighted total competition score")
    ax.set_title("Total score")
    for i, k in enumerate(order):
        ax.text(tot[k] + 0.004, i, f"{tot[k]:.4f}", va="center", fontsize=6.5)
    ax.set_xlim(0, max(max(tot.values()) * 1.20, BENCH_TOTAL * 1.20))
    ax.set_ylim(-0.6, len(order) - 0.1)

    fig.suptitle(
        "Step 4 -- molecular features, deep learning and out-of-fold stacking", fontsize=10
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"step4_performance_comparison.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {FIGURES / 'step4_performance_comparison.png'}")


def make_importance_figure() -> dict:
    """How much of the RDKit models' split gain actually goes to chemistry?

    This is the diagnostic that decides whether the molecular block is being used
    at all, as opposed to being present but ignored.
    """
    import lightgbm as lgb
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(WORKFLOW))
    import features as F

    rows, summary = [], {}
    for regime in REGIMES:
        p = MODELS4 / f"lgb_rdkit_delta__{regime}.txt"
        if not p.exists():
            continue
        b = lgb.Booster(model_file=str(p))
        names = b.feature_name()
        gain = np.asarray(b.feature_importance("gain"), dtype="float64")
        tot = gain.sum() or 1.0
        frac = gain / tot
        is_mol = np.array([n.startswith(("mol_", "fppca_")) for n in names])
        summary[regime] = {
            "molecular_share_of_gain": float(frac[is_mol].sum()),
            "tabular_share_of_gain": float(frac[~is_mol].sum()),
            "n_molecular_features": int(is_mol.sum()),
            "top_molecular": [
                {"feature": names[i], "gain_share": float(frac[i])}
                for i in np.argsort(-frac)
                if is_mol[i]
            ][:8],
        }
        for i in np.argsort(-frac)[:25]:
            rows.append(
                {
                    "regime": regime,
                    "feature": names[i],
                    "gain_share": float(frac[i]),
                    "is_molecular": bool(is_mol[i]),
                }
            )
    if not summary:
        log("  no RDKit models found; skipping importance figure")
        return {}

    imp = pd.DataFrame(rows)
    imp.to_csv(RESULTS / "step4_feature_importance.csv", index=False)

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    regs = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.5]})

    ax = axes[0]
    mol_share = [summary[r]["molecular_share_of_gain"] for r in regs]
    ax.bar(range(len(regs)), mol_share, color="#7B4EA8", lw=0)
    ax.set_xticks(range(len(regs)))
    ax.set_xticklabels(regs, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("share of total split gain")
    ax.set_title("Gain captured by the RDKit molecular block")
    for i, v in enumerate(mol_share):
        ax.text(i, v + 0.004, f"{v:.1%}", ha="center", fontsize=7)

    ax = axes[1]
    reg = "chem_novel" if "chem_novel" in summary else regs[0]
    sub = imp[imp.regime == reg].nlargest(14, "gain_share").iloc[::-1]
    cols = ["#7B4EA8" if m else "#3B6EA8" for m in sub.is_molecular]
    ax.barh(range(len(sub)), sub.gain_share, color=cols, lw=0)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub.feature, fontsize=6.5)
    ax.set_xlabel("share of total split gain")
    ax.set_title(f"Top features, '{reg}' specialist\n(purple = molecular, blue = tabular)")

    fig.suptitle("Step 4 -- does the model actually use the chemistry?", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"step4_feature_importance.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {FIGURES / 'step4_feature_importance.png'}")
    return summary


# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--figures-only",
        action="store_true",
        help="regenerate the Step-4 figures from the saved JSONs without re-predicting",
    )
    args = ap.parse_args()

    np.random.seed(SEED)
    log("=== Step 4.5: test predictions and verification ===")

    stack = json.loads((RESULTS / "step4_stacking_weights.json").read_text(encoding="utf-8"))
    if args.figures_only:
        log("figures-only mode: regenerating figures from saved scores")
        make_importance_figure()
        make_performance_figure(stack)
        log("=== figures regenerated ===")
        return

    roles = stack["roles"]
    Wt = stack["frozen_weights"]
    W = np.array([[Wt[r].get(k, 0.0) for k in roles] for r in REGIMES], dtype=float)
    log(f"frozen weights (roles {roles}):\n{json.dumps(Wt, indent=2)}")

    ctx = S4.load_context()
    meta, Y, D, C = ctx["meta"], ctx["Y"], ctx["D"], ctx["C"]
    masks, VS, S3, F = ctx["masks"], ctx["VS"], ctx["S3"], ctx["F"]
    proteins, enc = ctx["proteins"], ctx["enc"]
    train_mask = masks[VS.TRAIN_SPLIT]

    ev = load_module(WORKFLOW / "16_eval_gbdt.py", "ev16")

    log("loading the test cohort ...")
    te = S3.load_test(proteins, ctx["meta_all"], ctx["M_all"])
    meta_te, C_te = te["meta"], te["C"]
    D_te, Y_te = te["D"], te["Y"]
    n_te = len(meta_te)
    log(f"  test cohort: {n_te} treated samples x {len(proteins)} proteins")

    with np.errstate(all="ignore"):
        prot_mean_y = np.nanmean(Y[train_mask], axis=0).astype("float32")
        gmed = float(np.nanmedian(Y[train_mask]))
    prot_mean_y = np.where(np.isfinite(prot_mean_y), prot_mean_y, np.float32(gmed)).astype(
        "float32"
    )

    log("projecting the train-frozen abundance tables onto the test rows ...")
    Y_fb_te, _, _ = ev.project_train_abundance(
        meta_te, Y, meta, train_mask, VS.CTX_LEVELS, "abund_fallback_test", prot_mean_y
    )
    Y_bench_te, _, _ = ev.project_train_abundance(
        meta_te, Y, meta, train_mask, VS.CTX_LEVELS_BATCH, "abund_bench_test", prot_mean_y
    )

    # ---- members on the test cohort --------------------------------------
    mol = S4.load_mol_features("gbdt")
    ext = {"meta": meta_te, "C": C_te, "Y": None, "D": None}
    te_idx = np.arange(n_te)
    members: dict[str, np.ndarray] = {}

    if "gbdt_tab" in roles:
        fam = load_specialists("tab")
        raw, te_regimes = S4.predict_mol_families(
            {"gbdt_tab": fam}, enc, te_idx, meta_te, None, external=ext,
            chunk=250, label="test_tab",
        )
        members["gbdt_tab"] = raw["gbdt_tab"]
    if "gbdt_mol" in roles:
        fam = load_specialists("mol")
        raw, te_regimes = S4.predict_mol_families(
            {"gbdt_mol": fam}, enc, te_idx, meta_te, mol, external=ext,
            chunk=250, label="test_rdkit",
        )
        members["gbdt_mol"] = raw["gbdt_mol"]
    if "bench" in roles:
        members["bench"] = np.nan_to_num(
            (Y_bench_te - C_te).astype("float32"), nan=0.0, posinf=0.0, neginf=0.0
        )
    if "dl" in roles:
        log("predicting the test cohort with the MLP-ResNet ...")
        import torch

        dl = load_module(WORKFLOW / "24_train_deep_learning.py", "dl24")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fz = dl.SampleFeaturizer(meta, C, Y, train_mask)
        net = dl.build_model(fz).to(device)
        checkpoint = (MODELS4 / "dl_refit.pt").resolve()
        if not checkpoint.is_relative_to(MODELS4.resolve()):
            raise ValueError("Model checkpoint escaped the expected model directory")
        state = torch.load(
            checkpoint, map_location=device, weights_only=True
        )
        net.load_state_dict(state)
        members["dl"] = dl.predict(net, fz, meta_te, C_te, te_idx, device)

    for k in members:
        members[k] = np.nan_to_num(members[k], nan=0.0, posinf=0.0, neginf=0.0)

    te_route = pd.crosstab(meta_te["split_final"].to_numpy(), te_regimes)
    log("test regime routing vs split_final:\n" + te_route.to_string())

    # ---- apply the frozen weights ---------------------------------------
    coh = {"C_h": C_te, "regimes": te_regimes, "members": members}
    d_te = np.zeros_like(C_te)
    for ri, regime in enumerate(REGIMES):
        rows = np.flatnonzero(te_regimes == regime)
        if not len(rows):
            continue
        acc = np.zeros((len(rows), len(proteins)), dtype="float32")
        for ki, role in enumerate(roles):
            w = float(W[ri, ki])
            if w != 0.0 and role in members:
                acc += w * members[role][rows]
        d_te[rows] = acc

    Yh_te, Dh_te = S4.reconstruct(d_te, C_te, Y_fb_te)

    # ---- artefact integrity ---------------------------------------------
    checks: list[dict] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(ok), "detail": detail})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    req = (
        pd.read_parquet(WORKFLOW / "processed_delta_matrix_test.parquet", columns=["sample_ID"])[
            "sample_ID"
        ]
        .astype(str)
        .tolist()
    )
    ids = meta_te["sample_ID"].astype(str).tolist()
    chk("test predictions cover exactly the required sample_IDs, in order",
        ids == req, f"{len(ids)} samples")
    chk("prediction matrix has the full protein width",
        Yh_te.shape == (n_te, len(proteins)), f"shape {Yh_te.shape}")
    n_bad = int((~np.isfinite(Yh_te)).sum())
    chk("no non-finite value in the submitted abundance matrix", n_bad == 0,
        f"{n_bad} non-finite cells")
    lo, hi = float(Yh_te.min()), float(Yh_te.max())

    # The plan specified a [9, 33] log2 window. That window turns out to be a
    # convention, not a physical limit: the *measured* abundances themselves run
    # to 35.30 in train_val and 35.04 in test, with 647 and 299 observed cells
    # respectively above 33 (measured in 26a_probe_range.py). Gating on 33 would
    # therefore reject predictions that the data says are entirely ordinary --
    # the 7 Step-4 cells above 33 (0.00003% of 22.2 M) are all protein PDC1,
    # pyruvate decarboxylase, one of the most abundant proteins in yeast.
    #
    # The integrity gate is therefore the empirically defensible constraint --
    # predictions must lie inside the dynamic range actually observed in the
    # training data, which is the range available at submission time -- and the
    # plan's window is retained as an informational report rather than a gate.
    with np.errstate(all="ignore"):
        obs_lo = float(np.nanmin(Y[train_mask]))
        obs_hi = float(np.nanmax(Y[train_mask]))
    tol = 0.05
    chk(
        "predicted abundances lie within the observed train dynamic range",
        (lo >= obs_lo - tol) and (hi <= obs_hi + tol),
        f"predicted [{lo:.2f}, {hi:.2f}] vs observed train [{obs_lo:.2f}, {obs_hi:.2f}]",
    )
    n_out_plan = int(((Yh_te < 9.0) | (Yh_te > 33.0)).sum())
    log(
        f"  [info] plan's [9, 33] convention window: {n_out_plan} of {Yh_te.size} cells "
        f"({100 * n_out_plan / Yh_te.size:.6f}%) fall outside it; the measured truth also "
        f"exceeds 33 (observed train max {obs_hi:.2f}), so this window is reported, not gated"
    )

    # ---- write outputs ---------------------------------------------------
    out = pd.DataFrame(Yh_te, columns=proteins)
    out.insert(0, "sample_ID", ids)
    csv_p = RESULTS / "step4_test_predictions.csv"
    out.to_csv(csv_p, index=False, float_format="%.5f")
    log(f"wrote {csv_p.name} ({csv_p.stat().st_size / 1e6:.0f} MB)")
    out.to_parquet(DATA / "step4_test_predictions.parquet", index=False, compression="snappy")
    dlt = pd.DataFrame(Dh_te, columns=proteins)
    dlt.insert(0, "sample_ID", ids)
    dlt.to_parquet(DATA / "step4_test_delta_predictions.parquet", index=False,
                   compression="snappy")

    hdr = pd.read_csv(csv_p, nrows=3)
    chk("test-prediction CSV parses with the expected header",
        list(hdr.columns) == ["sample_ID"] + list(proteins), f"{hdr.shape[1]} columns")
    rt = pd.read_parquet(DATA / "step4_test_predictions.parquet")
    chk("CSV agrees with the parquet mirror",
        bool(np.allclose(rt[proteins].to_numpy("float32"), Yh_te, atol=1e-4)), "")

    # indicative (NOT official) fold-change score on the released test deltas
    sys.path.insert(0, str(WORKFLOW))
    import harness as H

    ind = H.module2_fold_change(D_te, Dh_te)

    # ---- success criteria ------------------------------------------------
    val_total = float(stack["val_total_at_frozen_weights"])
    crit = [
        {
            "name": "RDKit descriptors/fingerprints generated for 100% of chemical "
                    "perturbations without missing values",
            "met": None,
            "detail": "",
        },
        {
            "name": "final stacked ensemble total score > 0.4454 on the 5-module harness",
            "met": bool(val_total > BENCH_TOTAL),
            "detail": f"{val_total:.6f} vs {BENCH_TOTAL:.6f} "
                      f"(margin {val_total - BENCH_TOTAL:+.6f})",
        },
        {
            "name": "validated test prediction matrix exported without non-finite values",
            "met": n_bad == 0,
            "detail": f"{n_te} x {len(proteins)}, {n_bad} non-finite",
        },
    ]
    rdk = json.loads((RESULTS / "step4_rdkit_report.json").read_text(encoding="utf-8"))
    crit[0]["met"] = bool(
        rdk["n_unresolved"] == 0 and rdk["n_descriptor_missing_cells"] == 0
    )
    crit[0]["detail"] = (
        f"{rdk['n_resolved_as_molecule']}/{rdk['n_perturbation_labels'] - rdk['n_non_molecule_labels']} "
        f"molecular labels resolved, {rdk['n_unresolved']} unresolved, "
        f"{rdk['n_descriptor_missing_cells']} missing descriptor cells, "
        f"{rdk['mw_crosscheck']['n_flagged']} MW cross-checks flagged"
    )

    s1_gain = None
    try:
        ms = stack["module_scores"]
        if "val:gbdt_mol" in ms and "val:gbdt_tab" in ms:
            s1_gain = float(ms["val:gbdt_mol"]["m3_s1_chem"] - ms["val:gbdt_tab"]["m3_s1_chem"])
    except (KeyError, TypeError, ValueError):
        s1_gain = None
    crit.append(
        {
            "name": "RDKit-enhanced model improves S1 (val_chem_only) over the tabular model",
            "met": bool(s1_gain is not None and s1_gain > 0),
            "detail": f"m3_s1_chem delta = {s1_gain:+.6f}" if s1_gain is not None
            else "not measurable",
        }
    )

    imp_summary = make_importance_figure()
    make_performance_figure(stack)

    report = {
        "step": "4e_verification",
        "seed": SEED,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "artefact_integrity": {
            "n_checks": len(checks),
            "n_passed": sum(c["passed"] for c in checks),
            "all_passed": all(c["passed"] for c in checks),
            "note": "a failure here is a defect and raises",
            "checks": checks,
        },
        "success_criteria": {
            "n_criteria": len(crit),
            "n_met": sum(bool(c["met"]) for c in crit),
            "all_met": all(bool(c["met"]) for c in crit),
            "note": "a failure here is a scientific result, reported rather than raised",
            "criteria": crit,
        },
        "test_predictions": {
            "source": "Step-4 stacked ensemble with inner-OOF-frozen per-regime weights",
            "roles": roles,
            "frozen_weights": Wt,
            "n_samples": n_te,
            "n_proteins": len(proteins),
            "split_counts": {
                str(k): int(v) for k, v in meta_te["split_final"].value_counts().items()
            },
            "routing_vs_split": {
                str(k): {str(a): int(b) for a, b in v.items()}
                for k, v in te_route.to_dict(orient="index").items()
            },
            "y_pred_summary": {
                "mean": float(Yh_te.mean()),
                "sd": float(Yh_te.std()),
                "min": lo,
                "max": hi,
            },
            "dynamic_range_finding": {
                "observed_train_range": [obs_lo, obs_hi],
                "predicted_range": [lo, hi],
                "plan_convention_window": [9.0, 33.0],
                "n_cells_outside_plan_window": n_out_plan,
                "frac_cells_outside_plan_window": n_out_plan / float(Yh_te.size),
                "finding": (
                    "the plan's [9, 33] log2 window is a convention, not a physical bound: the "
                    "measured abundances run to 35.30 in train_val and 35.04 in test, with 647 "
                    "and 299 observed cells respectively above 33. The Step-4 predictions reach "
                    "33.04; the handful of cells above 33 are all protein PDC1 (pyruvate "
                    "decarboxylase), among the most abundant proteins in yeast, so they are "
                    "biologically plausible rather than artefactual. The integrity gate is "
                    "therefore the observed train dynamic range, and the plan window is "
                    "reported for reference only. No clipping is applied."
                ),
            },
            "delta_pred_summary": {
                "mean": float(np.nanmean(Dh_te)),
                "sd": float(np.nanstd(Dh_te)),
                "frac_defined": float(np.isfinite(Dh_te).mean()),
                "frac_abs_gt_1": float(np.nanmean(np.abs(Dh_te) > 1)),
            },
            "indicative_fold_change_pcc_per_sample_mean": float(
                np.nan_to_num(ind["pcc_per_sample_mean"], nan=0.0)
            ),
            "indicative_note": (
                "the released test delta matrix permits an indicative fold-change PCC; it is "
                "NOT the official score, whose mu_ctx / mu_drug baselines and test-side control "
                "matching are held by the organisers"
            ),
            "files": {
                "csv": str(csv_p),
                "parquet": str(DATA / "step4_test_predictions.parquet"),
                "delta_parquet": str(DATA / "step4_test_delta_predictions.parquet"),
            },
        },
        "molecular_feature_usage": imp_summary,
        "validation_summary": {
            "val_total_frozen_oof_weights": val_total,
            "benchmark_total": BENCH_TOTAL,
            "step3_best_total": stack.get("step3_best_total"),
            "inner_dev_total": stack.get("inner_dev_total_at_frozen_weights"),
            "val_total_val_tuned_OPTIMISTIC": stack.get("val_total_val_tuned_OPTIMISTIC"),
            "totals": stack["totals"],
        },
    }
    S4.write_json(RESULTS / "step4_verification_report.json", report)

    print("\n=== VERIFICATION ===")
    print(f"artefact integrity : {report['artefact_integrity']['n_passed']}"
          f"/{report['artefact_integrity']['n_checks']} passed")
    for c in crit:
        print(f"  [{'MET' if c['met'] else 'NOT MET'}] {c['name']}\n        {c['detail']}")

    if not report["artefact_integrity"]["all_passed"]:
        bad = [c["name"] for c in checks if not c["passed"]]
        raise AssertionError(f"artefact integrity failures: {bad}")
    log("=== Step 4 verification complete ===")


if __name__ == "__main__":
    main()
