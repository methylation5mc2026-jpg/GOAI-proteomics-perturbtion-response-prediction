"""Self-tests for workflow/harness.py.

Validates the scoring suite against independent reference implementations
(scipy / scikit-learn) and against analytically known answers, and pins the
degenerate-case behaviour that the real data will exercise heavily (all-NaN
slices, zero-variance slices, single-class DEP samples).

Run: ``uv run python workflow/11_test_harness.py``
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, r2_score

import harness as H

RNG = np.random.default_rng(42)
PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    """Record a single assertion outcome."""
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    else:
        FAIL.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}", flush=True)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    """True when both are NaN, or both finite and within ``tol``."""
    if not np.isfinite(a) and not np.isfinite(b):
        return True
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol)


# ---------------------------------------------------------------------------
print("\n[1] masked_pcc / masked_r2 vs scipy + sklearn", flush=True)

a = RNG.normal(size=400)
b = 0.6 * a + RNG.normal(size=400)
a_nan, b_nan = a.copy(), b.copy()
a_nan[RNG.random(400) < 0.20] = np.nan
b_nan[RNG.random(400) < 0.20] = np.nan
m = np.isfinite(a_nan) & np.isfinite(b_nan)

check("pooled PCC matches scipy.pearsonr on the finite intersection",
      approx(H.masked_pcc(a_nan, b_nan), pearsonr(a_nan[m], b_nan[m])[0], 1e-10),
      f"harness={H.masked_pcc(a_nan, b_nan):.12f} n={int(m.sum())}")

check("pooled R2 matches sklearn.r2_score on the finite intersection",
      approx(H.masked_r2(a_nan, b_nan), r2_score(a_nan[m], b_nan[m]), 1e-9),
      f"harness={H.masked_r2(a_nan, b_nan):.12f}")

# Per-axis against a slice-by-slice scipy loop.
A = RNG.normal(size=(40, 60))
B = 0.5 * A + RNG.normal(size=(40, 60))
A[RNG.random(A.shape) < 0.15] = np.nan
B[RNG.random(B.shape) < 0.15] = np.nan

for axis, label in [(1, "per-sample"), (0, "per-protein")]:
    got = np.asarray(H.masked_pcc(A, B, axis=axis))
    ref = []
    for i in range(A.shape[1 - axis]):
        x, y = (A[i], B[i]) if axis == 1 else (A[:, i], B[:, i])
        mm = np.isfinite(x) & np.isfinite(y)
        if mm.sum() < H.MIN_N or np.var(x[mm]) == 0 or np.var(y[mm]) == 0:
            ref.append(np.nan)
        else:
            ref.append(pearsonr(x[mm], y[mm])[0])
    ref = np.asarray(ref)
    ok = all(approx(g, r, 1e-9) for g, r in zip(got, ref))
    check(f"{label} PCC vector matches a scipy loop", ok,
          f"n_slices={got.size}, n_defined={int(np.isfinite(got).sum())}")

    got_r2 = np.asarray(H.masked_r2(A, B, axis=axis))
    ref_r2 = []
    for i in range(A.shape[1 - axis]):
        x, y = (A[i], B[i]) if axis == 1 else (A[:, i], B[:, i])
        mm = np.isfinite(x) & np.isfinite(y)
        if mm.sum() < H.MIN_N or np.var(x[mm]) == 0:
            ref_r2.append(np.nan)
        else:
            ref_r2.append(r2_score(x[mm], y[mm]))
    ok = all(approx(g, r, 1e-8) for g, r in zip(got_r2, np.asarray(ref_r2)))
    check(f"{label} R2 vector matches an sklearn loop", ok)

# ---------------------------------------------------------------------------
print("\n[2] degenerate and adversarial cases", flush=True)

x = RNG.normal(size=50)
check("identical vectors give PCC = 1", approx(H.masked_pcc(x, x), 1.0, 1e-12))
check("negated vector gives PCC = -1", approx(H.masked_pcc(x, -x), -1.0, 1e-12))
check("identical vectors give R2 = 1", approx(H.masked_r2(x, x), 1.0, 1e-12))

const = np.full(50, 3.0)
check("zero-variance prediction gives NaN PCC (not 0.0)",
      np.isnan(H.masked_pcc(x, const)))
check("zero-variance truth gives NaN R2 (not 0.0)",
      np.isnan(H.masked_r2(const, x)))

short = np.array([1.0, 2.0])
check(f"fewer than MIN_N={H.MIN_N} pairs gives NaN",
      np.isnan(H.masked_pcc(short, short * 2)))

allnan = np.full(50, np.nan)
check("all-NaN input gives NaN, no exception", np.isnan(H.masked_pcc(x, allnan)))

disjoint_a = np.array([1.0, 2.0, 3.0, np.nan, np.nan, np.nan])
disjoint_b = np.array([np.nan, np.nan, np.nan, 1.0, 2.0, 3.0])
check("disjoint NaN patterns give NaN (empty intersection)",
      np.isnan(H.masked_pcc(disjoint_a, disjoint_b)))

try:
    H.masked_pcc(np.zeros(5), np.zeros(6))
    check("shape mismatch raises ValueError", False, "no exception raised")
except ValueError:
    check("shape mismatch raises ValueError", True)

check("_clip01 floors negative correlation at 0", H._clip01(-0.7) == 0.0)
check("_clip01 maps NaN to 0", H._clip01(float("nan")) == 0.0)
check("_clip01 caps at 1", H._clip01(1.4) == 1.0)

# R2 must be the residual-based definition, not squared correlation:
# a perfectly correlated but mis-scaled prediction has R2 < 0.
biased = x * 5.0 + 100.0
check("R2 penalises scale/offset error even at PCC = 1",
      approx(H.masked_pcc(x, biased), 1.0, 1e-12) and H.masked_r2(x, biased) < 0,
      f"pcc=1.0, r2={H.masked_r2(x, biased):.3f}")

# ---------------------------------------------------------------------------
print("\n[3] module4_dep vs sklearn and hand-computed values", flush=True)

# Hand-built single sample: 6 proteins, positives are |delta_true| > 1.
dt = np.array([[3.0, -2.0, 1.5, 0.2, -0.1, 0.5]])     # positives: idx 0,1,2 -> n_pos=3
dp = np.array([[2.5, -0.4, 4.0, 1.4, 0.0, 0.1]])      # |dp|>1: idx 0,2,3
dep = H.module4_dep(dt, dp)
# tp = {0,2} = 2 ; fp = {3} = 1 ; fn = {1} = 1
check("DEP precision = tp/(tp+fp) = 2/3", approx(dep["precision_macro"], 2 / 3, 1e-12),
      f"got {dep['precision_macro']:.6f}")
check("DEP recall = tp/n_pos = 2/3", approx(dep["recall_macro"], 2 / 3, 1e-12))
check("DEP F1 = 2*2/(2*2+1+1) = 2/3", approx(dep["f1_macro"], 2 / 3, 1e-12))
check("DEP n_positives_total = 3", dep["n_positives_total"] == 3)
# Ranking by |dp| desc: idx2(4.0), idx0(2.5), idx3(1.4), idx1(0.4), idx5(0.1), idx4(0.0)
# Top-K with K=n_pos=3 -> {2,0,3} contains 2 true positives -> 2/3
check("DEP Recall@K (K=n_pos) = 2/3", approx(dep["recall_at_k_macro"], 2 / 3, 1e-12))
check("DEP AUPRC matches sklearn average_precision_score",
      approx(dep["auprc_macro"], average_precision_score(
          np.abs(dt[0]) > 1.0, np.abs(dp[0])), 1e-12),
      f"got {dep['auprc_macro']:.6f}")

# Perfect ranking -> AUPRC and Recall@K both 1.
dep_perfect = H.module4_dep(dt, dt)
check("perfect DEP prediction gives precision/recall/F1/AUPRC/Recall@K = 1",
      all(approx(dep_perfect[k], 1.0, 1e-12) for k in
          ["precision_macro", "recall_macro", "f1_macro", "auprc_macro", "recall_at_k_macro"]))

# A sample with no positives contributes nothing but must not crash or bias.
dt2 = np.vstack([dt, np.array([[0.1, 0.2, -0.3, 0.0, 0.1, -0.2]])])
dp2 = np.vstack([dp, np.array([[0.1, 0.2, -0.3, 0.0, 0.1, -0.2]])])
dep2 = H.module4_dep(dt2, dp2)
check("samples with zero positives are excluded, not scored as 0",
      dep2["precision_n_defined"] == 1 and approx(dep2["precision_macro"], 2 / 3, 1e-12),
      f"n_defined={dep2['precision_n_defined']}")

# NaN cells must be dropped from the DEP mask.
dt3, dp3 = dt.copy(), dp.copy()
dt3[0, 1] = np.nan                        # removes the fn -> recall becomes 2/2
dep3 = H.module4_dep(dt3, dp3)
check("NaN cells are excluded from DEP counts", approx(dep3["recall_macro"], 1.0, 1e-12),
      f"recall={dep3['recall_macro']:.6f}")

# ---------------------------------------------------------------------------
print("\n[4] module3_residual semantics", flush=True)

d_true = RNG.normal(size=(30, 80))
mu = RNG.normal(size=(30, 80))
res_perfect = H.module3_residual(d_true, d_true, mu)
check("perfect prediction gives residual PCC = 1",
      approx(res_perfect["resid_pcc_pooled"], 1.0, 1e-9))

# A model that outputs exactly the baseline has zero residual variance -> NaN,
# which floors to 0. This is the intended trivial-baseline lower bound.
res_trivial = H.module3_residual(d_true, mu, mu)
check("predicting the baseline itself gives an undefined (NaN) residual PCC",
      np.isnan(res_trivial["resid_pcc_pooled"]),
      "zero residual variance in the prediction")
check("undefined residual PCC floors to 0 in the score",
      H._clip01(res_trivial["resid_pcc_pooled"]) == 0.0)

# Residual is invariant to adding any function of mu to both sides.
res_a = H.module3_residual(d_true, 0.5 * d_true, mu)
res_b = H.module3_residual(d_true + 0.0, 0.5 * d_true, mu)
check("residual PCC is deterministic / reproducible",
      approx(res_a["resid_pcc_pooled"], res_b["resid_pcc_pooled"], 0.0))

check("residual metric names honour the prefix argument",
      "resid_ctx_pcc_pooled" in H.module3_residual(d_true, d_true, mu, prefix="resid_ctx"))

# A row of all-NaN baseline must not poison other rows.
mu_gap = mu.copy()
mu_gap[0] = np.nan
res_gap = H.module3_residual(d_true, d_true, mu_gap)
check("all-NaN baseline row drops out without poisoning the rest",
      approx(res_gap["resid_pcc_pooled"], 1.0, 1e-9)
      and res_gap["resid_pcc_per_sample_n_defined"] == 29,
      f"n_defined={res_gap['resid_pcc_per_sample_n_defined']}")

# ---------------------------------------------------------------------------
print("\n[5] compute_competition_score end-to-end", flush=True)

check("module weights sum to exactly 1.0",
      approx(sum(H.MODULE_WEIGHTS.values()), 1.0, 1e-12),
      f"sum={sum(H.MODULE_WEIGHTS.values())}")
for mod, sub in H.SUBMETRIC_WEIGHTS.items():
    check(f"sub-weights of {mod} sum to 1.0", approx(sum(sub.values()), 1.0, 1e-12))

n, p = 200, 300
splits = np.array(["val_chem_only"] * 80 + ["val_strain_only"] * 70 +
                  ["val_both"] * 30 + ["val_time"] * 20)
meta = pd.DataFrame({"split_final": splits})

C = RNG.normal(loc=20, scale=2, size=(n, p)).astype("float32")
D_true = (RNG.normal(size=(n, p)) * 1.2).astype("float32")   # gives real DEPs
Y_true = (C + D_true).astype("float32")
mu_ctx = (RNG.normal(size=(n, p)) * 0.3).astype("float32")
mu_drug = (RNG.normal(size=(n, p)) * 0.3).astype("float32")
# Puncture with NaN at the observed ~29% missing rate of the real Delta matrix.
nan_mask = RNG.random((n, p)) < 0.29
Y_nan, D_nan = Y_true.copy(), D_true.copy()
Y_nan[nan_mask] = np.nan
D_nan[nan_mask] = np.nan

perfect = H.compute_competition_score(Y_nan, Y_nan, D_nan, D_nan, meta,
                                      mu_ctx=mu_ctx, mu_drug=mu_drug, verbose=False)
check("a perfect prediction scores exactly 1.0",
      approx(perfect["total_score"], 1.0, 1e-6),
      f"total={perfect['total_score']:.9f}")
check("every module scores 1.0 for a perfect prediction",
      all(approx(v, 1.0, 1e-6) for v in perfect["module_scores"].values()),
      str({k: round(v, 6) for k, v in perfect["module_scores"].items()}))
check("no warnings raised on a fully specified call", perfect["warnings"] == [],
      str(perfect["warnings"]))

# Pure noise must land near zero, and can never be negative.
noise = RNG.normal(size=(n, p)).astype("float32")
noise[nan_mask] = np.nan
rand_res = H.compute_competition_score(Y_nan, noise + 20, D_nan, noise, meta,
                                       mu_ctx=mu_ctx, mu_drug=mu_drug, verbose=False)
check("an uninformative prediction scores >= 0 and well below 1",
      0.0 <= rand_res["total_score"] < 0.35,
      f"total={rand_res['total_score']:.6f}")

# Anti-correlated prediction must not be rewarded.
anti = H.compute_competition_score(Y_nan, -Y_nan, D_nan, -D_nan, meta,
                                   mu_ctx=mu_ctx, mu_drug=mu_drug, verbose=False)
check("an anti-correlated prediction scores 0 on the correlation modules",
      all(anti["module_scores"][k] == 0.0 for k in
          ["m2_fold_change", "m3_s1_chem", "m3_s2_strain"]),
      str({k: round(v, 4) for k, v in anti["module_scores"].items()}))

check("weighted contributions reproduce the total",
      approx(sum(perfect["module_weighted_contributions"].values()),
             perfect["total_score"], 1e-12))

# Missing residual baselines must degrade loudly, not silently.
no_mu = H.compute_competition_score(Y_nan, Y_nan, D_nan, D_nan, meta, verbose=False)
check("omitting mu_ctx/mu_drug emits warnings and zeroes those terms",
      len(no_mu["warnings"]) == 6 and no_mu["module_scores"]["m3_s1_chem"] == 0.0,
      f"{len(no_mu['warnings'])} warnings")

# Shape / alignment validation.
for bad, why in [
    ((Y_nan, Y_nan[:, :10], D_nan, D_nan, meta), "y_pred column mismatch"),
    ((Y_nan, Y_nan, D_nan[:5], D_nan, meta), "delta_true row mismatch"),
    ((Y_nan, Y_nan, D_nan, D_nan, meta.iloc[:5]), "metadata row mismatch"),
    ((Y_nan, Y_nan, D_nan, D_nan, pd.DataFrame({"x": splits})), "missing split_final"),
]:
    try:
        H.compute_competition_score(*bad, verbose=False)
        check(f"raises on {why}", False, "no exception raised")
    except ValueError:
        check(f"raises on {why}", True)

# No NaN may survive into the reported scores.
check("no NaN in total or module scores",
      np.isfinite(rand_res["total_score"])
      and all(np.isfinite(v) for v in rand_res["module_scores"].values()))

# ---------------------------------------------------------------------------
print("\n[6] score_sensitivity and flatten_scores", flush=True)

sens = H.score_sensitivity(perfect)
check("all conventions score a perfect prediction at 1.0",
      all(approx(v["total_score"], 1.0, 1e-6) for v in sens.values()),
      str({k: round(v["total_score"], 6) for k, v in sens.items()}))
sens_noise = H.score_sensitivity(rand_res)
check("all conventions score an uninformative prediction near 0",
      all(0.0 <= v["total_score"] < 0.4 for v in sens_noise.values()),
      str({k: round(v["total_score"], 4) for k, v in sens_noise.items()}))

rows = H.flatten_scores("unit_test", rand_res)
df = pd.DataFrame(rows)
check("flatten_scores emits a TOTAL row plus one MODULE_SCORE row per module",
      (df["module"] == "TOTAL").sum() == 1
      and (df["metric"] == "MODULE_SCORE").sum() == len(H.MODULE_WEIGHTS),
      f"{len(df)} rows total")

# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print(f"harness self-tests: {len(PASS)} passed, {len(FAIL)} failed")
print("=" * 68)
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("All harness self-tests PASSED.")
