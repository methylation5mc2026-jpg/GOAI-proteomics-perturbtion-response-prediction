"""Step 3d: how the competition score depends on the boosting-round count.

Why this exists
---------------
The pre-registered selection (``15_train_gbdt.py``) chose 500 dev iterations for
every regime, i.e. the top of its ladder, and the resulting 600-round models
score *below* an otherwise identical 60-round model on Modules 1 and S1.  Two
things changed between those runs -- regime masking and the round count -- so the
comparison cannot attribute the difference.  This script isolates the round count
by truncating the *already trained* boosters at prediction time (LightGBM
``num_iteration``, XGBoost ``iteration_range``, CatBoost ``ntree_end``), so no
model is refitted and nothing else varies.

Reading the result
------------------
The curve is a **sensitivity analysis, not a selection**.  Its x-axis is scored on
the very ``val_*`` rows that the headline number is reported on, so picking the
argmax here and quoting it as the model's score would be selection on the test
set.  What it legitimately establishes is *whether the inner-dev round selection
was miscalibrated*, and by how much -- which is exactly the information Step 4
needs to build a better selection procedure.

Both numbers are exported and both are reported: the pre-registered 600-round
score (the honest headline) and the best-on-val score (an optimistic upper bound
on what better round selection could buy).
"""

from __future__ import annotations

from repo_paths import WORKFLOW_DIR

import json
import sys
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(WORKFLOW_DIR))

for _msg in ("Mean of empty slice", "All-NaN slice encountered",
             "invalid value encountered", "Degrees of freedom <= 0"):
    warnings.filterwarnings("ignore", message=_msg)

import features as F  # noqa: E402
import harness as H  # noqa: E402
import step3_data as S3  # noqa: E402
import validation_splits as VS  # noqa: E402
from common import FIGURES, RESULTS, SEED  # noqa: E402

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "eval16", str(WORKFLOW_DIR / "16_eval_gbdt.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

#: Truncation points.  600 is the count the pre-registered selection shipped.
ROUNDS = [25, 50, 100, 200, 400, 600]
PREREG_ROUNDS = 600

T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def truncated_predict(model, lib: str, X: pd.DataFrame, n: int, cache: dict) -> np.ndarray:
    """Predict using only the first ``n`` trees of an already-fitted booster."""
    if lib == "lgb":
        return model.predict(X, num_iteration=n)
    if lib == "xgb":
        import xgboost as xgb
        if "dmat" not in cache:
            cache["dmat"] = xgb.DMatrix(X, enable_categorical=True, nthread=24)
        return model.predict(cache["dmat"], iteration_range=(0, n))
    if "cframe" not in cache:
        cache["cframe"] = E._cat_frame(X)
    return model.predict(cache["cframe"], ntree_end=n)


def main() -> None:
    log("=== Step 3d: boosting-round sensitivity (no refitting) ===")
    log(f"truncations: {ROUNDS} | pre-registered: {PREREG_ROUNDS}")

    tv = S3.load_train_val()
    meta, Y, D, C, masks = tv["meta"], tv["Y"], tv["D"], tv["C"], tv["masks"]
    C_harness = tv["C_harness"]
    train_mask, eval_mask = masks[VS.TRAIN_SPLIT], masks["all_val"]

    mu_ctx, mu_drug, _ = VS.build_residual_baselines(meta, D, masks)
    Y_ctx_fb, _, _ = VS.frozen_delta_baseline(meta, Y, train_mask, VS.CTX_LEVELS,
                                              "abund_ctx_fallback")
    with np.errstate(all="ignore"):
        prot_mean_y = np.nanmean(Y[train_mask], axis=0).astype("float32")
        gmed = float(np.nanmedian(Y[train_mask]))
    prot_mean_y = np.where(np.isfinite(prot_mean_y), prot_mean_y,
                           np.float32(gmed)).astype("float32")
    y_fallback = np.where(np.isfinite(Y_ctx_fb), Y_ctx_fb,
                          np.broadcast_to(prot_mean_y, Y.shape)).astype("float32")

    enc = F.EncoderSet(meta, Y, D, C, train_mask, n_folds=F.N_FOLDS, seed=SEED)
    families = E.load_models()
    eval_idx = np.flatnonzero(eval_mask)
    regimes = E.regimes_for_samples(enc, meta, eval_idx)
    n, p = len(eval_idx), enc.p

    # One design pass; every family and every truncation scores the same blocks.
    log(f"predicting {len(families)} families x {len(ROUNDS)} truncations on "
        f"{n} samples (single design pass) ...")
    raw = {(fam, r): np.empty((n, p), dtype="float32")
           for fam in families for r in ROUNDS}
    done = 0
    for regime in F.REGIMES:
        pos_all = np.flatnonzero(regimes == regime)
        if not len(pos_all):
            continue
        for s in range(0, len(pos_all), 250):
            pos = pos_all[s:s + 250]
            X, _, _, _ = enc.build_block(eval_idx[pos])
            cache: dict = {}
            for fam, by_regime in families.items():
                model, lib, _ = by_regime[regime]
                for r in ROUNDS:
                    yp = truncated_predict(model, lib, X, r, cache)
                    raw[(fam, r)][pos] = yp.astype("float32").reshape(len(pos), p)
            del X, cache
            done += len(pos)
            log(f"  [pred] {done}/{n} samples ({regime})")

    meta_eval = meta.loc[eval_mask].reset_index(drop=True)
    Y_ev, D_ev, C_ev = Y[eval_mask], D[eval_mask], C_harness[eval_mask]
    mu_ctx_ev, mu_drug_ev, y_fb_ev = mu_ctx[eval_mask], mu_drug[eval_mask], y_fallback[eval_mask]

    rows: list[dict] = []
    for r in ROUNDS:
        raw_r = {fam: raw[(fam, r)] for fam in families}
        preds = E.assemble_predictions(families, raw_r, C_ev, y_fb_ev)
        for nm, (Yh, Dh) in preds.items():
            res = H.compute_competition_score(Y_ev, Yh, D_ev, Dh, meta_eval,
                                              mu_ctx=mu_ctx_ev, mu_drug=mu_drug_ev,
                                              verbose=False)
            rows.append({"model": nm, "n_rounds": r,
                         "total_score": res["total_score"],
                         **{k: v for k, v in res["module_scores"].items()}})
            log(f"  [score] {nm:12s} @{r:4d} rounds -> {res['total_score']:.6f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "gbdt_round_sensitivity.csv", index=False)
    log(f"wrote results/gbdt_round_sensitivity.csv ({len(df)} rows)")

    sc = json.loads((RESULTS / "gbdt_model_scores.json").read_text(encoding="utf-8"))
    bench = float(sc["success_gate"]["benchmark_total_score"])

    pre = df[df["n_rounds"] == PREREG_ROUNDS].nlargest(1, "total_score").iloc[0]
    top = df.nlargest(1, "total_score").iloc[0]
    log(f"pre-registered ({PREREG_ROUNDS} rounds): best = {pre['model']} "
        f"{pre['total_score']:.6f} (benchmark {bench:.6f})")
    log(f"best over the whole grid:      {top['model']} @{int(top['n_rounds'])} "
        f"= {top['total_score']:.6f}  <-- SELECTED ON val_*, optimistic")

    piv = df.pivot(index="n_rounds", columns="model", values="total_score")
    print("\nTotal score vs boosting rounds:")
    print(piv.round(4).to_string(), flush=True)

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step": "3d_round_sensitivity",
        "method": ("already-trained boosters truncated at prediction time (LightGBM "
                   "num_iteration / XGBoost iteration_range / CatBoost ntree_end); no model "
                   "is refitted, so the round count is the only thing that varies"),
        "interpretation": (
            "This curve is scored on the same val_* rows the headline score is reported on. "
            "Its argmax is therefore NOT a legitimate model score -- quoting it would be "
            "selection on the evaluation set. What it does establish is that the inner-dev "
            "round selection was miscalibrated, and it bounds how much a better selection "
            "procedure could recover."),
        "rounds": ROUNDS,
        "prereg_rounds": PREREG_ROUNDS,
        "benchmark_total_score": bench,
        "prereg_best": {"model": str(pre["model"]), "n_rounds": PREREG_ROUNDS,
                        "total_score": float(pre["total_score"]),
                        "beats_benchmark": bool(pre["total_score"] > bench)},
        "grid_best_optimistic": {"model": str(top["model"]),
                                 "n_rounds": int(top["n_rounds"]),
                                 "total_score": float(top["total_score"]),
                                 "beats_benchmark": bool(top["total_score"] > bench),
                                 "caveat": "selected on val_*; upper bound, not a fair score"},
        "curve": E.jsonable(df.to_dict(orient="records")),
    }
    (RESULTS / "gbdt_round_sensitivity.json").write_text(
        json.dumps(E.jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    log("wrote results/gbdt_round_sensitivity.json")

    make_figure(df, bench)
    log(f"=== Step 3d complete in {time.time() - T0:.1f}s ===")


def _log_ticks(ax) -> None:
    """Log x-axis labelled at the sampled round counts only.

    Matplotlib's default log minor ticks (3x10^1, 4x10^1, ...) collide at this
    figure width and label points that were never evaluated.
    """
    ax.set_xscale("log")
    ax.set_xticks(ROUNDS)
    ax.set_xticklabels([str(r) for r in ROUNDS], fontsize=7.5)
    ax.minorticks_off()
    ax.set_xlabel("boosting rounds used at prediction time (log scale)")


def make_figure(df: pd.DataFrame, bench: float) -> None:
    """Total and per-module score against the boosting-round count."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))

    ax = axes[0]
    for nm, g in df.groupby("model"):
        g = g.sort_values("n_rounds")
        ax.plot(g["n_rounds"], g["total_score"], marker="o", ms=3.5, lw=1.2, label=nm)
    ax.axhline(bench, color="k", ls="--", lw=1.0)
    ax.text(df["n_rounds"].max(), bench, f" benchmark {bench:.4f}", fontsize=7,
            va="bottom", ha="right")
    ax.axvline(PREREG_ROUNDS, color="grey", ls=":", lw=1.0)
    ax.text(PREREG_ROUNDS, ax.get_ylim()[0], " pre-registered", fontsize=6.5,
            rotation=90, va="bottom")
    _log_ticks(ax)
    ax.set_ylabel("weighted total score")
    ax.set_title("Total competition score vs model capacity\n"
                 "no configuration reaches the benchmark", fontsize=9.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    ax.grid(lw=0.3, alpha=0.5)

    ax = axes[1]
    ref = "xgb_delta" if (df["model"] == "xgb_delta").any() else df["model"].iloc[0]
    g = df[df["model"] == ref].sort_values("n_rounds")
    for mod in H.MODULE_WEIGHTS:
        ax.plot(g["n_rounds"], g[mod], marker="o", ms=3.5, lw=1.2,
                label=f"{mod} ({H.MODULE_WEIGHTS[mod]:.0%})")
    _log_ticks(ax)
    ax.set_ylabel("module score (0-1)")
    ax.set_title(f"Per-module score vs capacity -- {ref}\n"
                 "modules diverge: S2 and Time want capacity, abundance and S1 do not",
                 fontsize=9.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    ax.grid(lw=0.3, alpha=0.5)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"gbdt_round_sensitivity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [fig] figures/gbdt_round_sensitivity.png|pdf", flush=True)


if __name__ == "__main__":
    main()
