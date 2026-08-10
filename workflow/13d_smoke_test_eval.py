"""Smoke test the pure functions of 16_eval_gbdt.py on synthetic inputs.

Checks the reconstruction algebra, the ensemble assembly and the bootstrap
machinery without loading any real model, so logic errors surface in seconds
rather than after a 25-minute prediction pass.
"""

from __future__ import annotations

from repo_paths import WORKFLOW_DIR

import importlib.util
import sys
from pathlib import Path

import numpy as np

WF = WORKFLOW_DIR
sys.path.insert(0, str(WF))

# The evaluation script is named with a numeric prefix, so import it by path.
spec = importlib.util.spec_from_file_location("eval16", WF / "16_eval_gbdt.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

RNG = np.random.default_rng(3)


def main() -> None:
    n, p = 7, 11
    C = RNG.normal(12, 1, (n, p)).astype("float32")
    C[RNG.random((n, p)) < 0.2] = np.nan          # undetected control cells
    fb = RNG.normal(12, 1, (n, p)).astype("float32")
    pred = RNG.normal(0, 0.4, (n, p)).astype("float32")

    # --- 1. delta formulation ---------------------------------------------
    yh, dh = E.reconstruct("delta", pred, C, fb)
    assert np.array_equal(dh, pred), "delta formulation must pass the prediction through"
    ok = np.isfinite(C)
    assert np.allclose(yh[ok], (C + pred)[ok]), "y_hat != C + Delta where C is defined"
    assert np.isfinite(yh).all(), "fallback failed to fill the undetected-anchor cells"
    assert np.allclose(yh[~ok], (fb + pred)[~ok]), "fallback did not use y_fallback + pred"
    print("  ok: delta formulation reconstructs y_hat with a finite fallback")

    # --- 2. abs formulation -----------------------------------------------
    absy = RNG.normal(12, 1, (n, p)).astype("float32")
    yh2, dh2 = E.reconstruct("abs", absy, C, fb)
    assert np.array_equal(yh2, absy), "abs formulation must pass the prediction through"
    assert np.allclose(dh2[ok], (absy - C)[ok]), "Delta != y_hat - C"
    assert (~np.isfinite(dh2[~ok])).all(), "Delta must stay undefined where C is undefined"
    print("  ok: abs formulation derives Delta and leaves it undefined without an anchor")

    # --- 3. the anchor cancels in Delta -----------------------------------
    # This is why a model can look strong on Module 1 and collapse on Module 2:
    # whatever anchor is used, it cancels when the harness forms Delta.
    yh3, dh3 = E.reconstruct("delta", pred, C, fb)
    dd = yh3 - C
    both = np.isfinite(dd) & np.isfinite(pred)
    assert np.allclose(dd[both], pred[both], atol=1e-5), \
        "harness-side Delta_pred = y_hat - C must recover the model's own Delta"
    print("  ok: harness-side Delta_pred = y_hat - C recovers the model output exactly")

    # --- 4. ensemble assembly ---------------------------------------------
    fams = {
        "a_delta": {"full": (None, "lgb", "delta")},
        "b_delta": {"full": (None, "xgb", "delta")},
        "c_abs": {"full": (None, "cat", "abs")},
    }
    raw = {"a_delta": pred, "b_delta": pred * 2, "c_abs": absy}
    preds = E.assemble_predictions(fams, raw, C, fb)
    assert set(preds) >= {"a_delta", "b_delta", "c_abs", "ens_delta", "ens_all"}, \
        f"missing ensembles: {sorted(preds)}"
    assert "ens_abs" not in preds, "ens_abs must not be formed from a single abs member"
    assert np.allclose(preds["ens_delta"][1], (pred + pred * 2) / 2), \
        "ens_delta is not the mean of the delta members' raw output"
    # ens_all must average the *reconstructed* deltas, never the raw outputs
    # (one member's raw output is an abundance, not a fold-change).
    exp = np.nanmean(np.stack([preds[k][1] for k in ("a_delta", "b_delta", "c_abs")]), axis=0)
    got = preds["ens_all"][1]
    m = np.isfinite(exp) & np.isfinite(got)
    assert np.allclose(got[m], np.mean(np.stack(
        [preds[k][1] for k in ("a_delta", "b_delta", "c_abs")]), axis=0)[m], atol=1e-5), \
        "ens_all must average reconstructed Delta, not raw model output"
    print("  ok: ensembles average in Delta space and skip single-member groups")

    # --- 5. bootstrap ------------------------------------------------------
    v = RNG.normal(0.4, 0.1, 200)
    ci = E.boot_ci(v, n_boot=500, seed=1)
    assert ci["ci_lo"] < ci["mean"] < ci["ci_hi"], "CI does not bracket the mean"
    assert abs(ci["mean"] - v.mean()) < 1e-12, "reported mean is not the sample mean"
    assert E.boot_ci(np.array([np.nan, 1.0]))["n"] == 1, "NaN handling wrong"
    assert not np.isfinite(E.boot_ci(np.array([1.0, 2.0]))["mean"]), \
        "a 2-sample vector should yield an undefined CI"
    print(f"  ok: boot_ci brackets the mean ({ci['ci_lo']:.4f} < {ci['mean']:.4f} "
          f"< {ci['ci_hi']:.4f})")

    # A clearly better vector must give a positive paired difference whose CI
    # excludes zero; two identical vectors must not.
    a = v + 0.05
    pb = E.paired_boot(a, v, n_boot=500, seed=1)
    assert pb["ci_lo"] > 0 and pb["p_two_sided"] < 0.05, f"paired bootstrap too weak: {pb}"
    assert abs(pb["mean_diff"] - 0.05) < 1e-10
    same = E.paired_boot(v, v.copy(), n_boot=500, seed=1)
    assert abs(same["mean_diff"]) < 1e-12 and same["ci_lo"] == same["ci_hi"] == 0.0, \
        f"identical vectors should give a zero-width interval at 0: {same}"
    print(f"  ok: paired_boot detects a +0.05 shift (p={pb['p_two_sided']:.4f}) and "
          f"reports zero for identical inputs")

    print("\nALL EVAL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
