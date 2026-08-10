"""Exact fast evaluation of the harness total for *protein-cluster* blend weights.

The extension, and why it costs nothing extra
---------------------------------------------
:mod:`step4_fastscore` evaluates ``P = B + sum_k w_k F_k`` with one scalar weight
per (regime, role). Step 5 needs one weight per (regime, role, **protein
cluster**)::

    P[i, j] = B[i, j] + sum_k W[regime(i), k, cluster(j)] * F_k[i, j]

The naive route is to treat each (role, cluster) pair as a separate pseudo-member,
which would inflate the Gram tensor from ``(K+1)^2`` to ``(K*C+1)^2`` entries --
for K=5 roles and C=8 clusters, a 41x41 Gram instead of 6x6, i.e. ~47x the
precompute.

That is unnecessary, because **clusters partition the protein axis**. Pseudo-member
``(k, c)`` is zero on every column outside cluster ``c``, so every cross-cluster
Gram entry ``sum(M * F_{k,c} * F_{l,c'})`` with ``c != c'`` is *identically zero*.
The Gram is block-diagonal in the cluster index. So instead of one big block per
regime, this module builds one small block per **(regime, cluster)** pair on the
original ``K+1`` members, restricted to that cluster's columns. Each protein is
visited exactly once, so the total precompute is the same as Step 4's -- the
weight space grows by a factor of ``C`` for free.

Accumulating the blocks back together
-------------------------------------
The two reduction axes recombine differently, and getting this backwards is the
one way the extension could silently stop being exact:

* ``axis = 0`` (one value per protein): a protein sits in exactly one cluster but
  receives rows from **every** regime, so its moments are summed over the regime
  blocks of its own cluster and scattered to its column position.
* ``axis = 1`` (one value per sample): a row sits in exactly one regime but spans
  **every** cluster, so its moments are summed over the cluster blocks of its own
  regime and scattered to its row position.
* pooled: sum everything.

Because a protein's per-protein moments accumulate across regimes with *different*
weight vectors, the per-protein Pearson/R^2 of a cluster-weighted blend is **not**
a per-cluster quantity -- which is exactly why this has to be accumulated rather
than computed cluster by cluster and averaged.

Scope, stated plainly
---------------------
Modules 1, 2 and 3 (0.95 of the total weight) are reproduced exactly: the same
formulae, the same ``MIN_N`` and zero-variance NaN rules, the same ``clip01``,
float64 accumulation throughout. Module 4 (0.05) thresholds ``|Delta| > 1`` and is
not a polynomial in the weights, so it is held constant during the search and
every shortlisted candidate is re-scored with the real harness. **No reported
number comes from this module.** :meth:`ClusterFastScorer.validate` asserts
agreement with the real harness on random cluster-weight tensors, including
degenerate ones, before the optimiser is permitted to use it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WORKFLOW = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKFLOW))

from step4_fastscore import _Block, _clip01, _mean_finite  # noqa: E402

MIN_N = 3


class _CTask:
    """One (truth, offset, row-subset, axis) configuration over regime x cluster."""

    def __init__(self, T, B, members: dict, roles: list[str], regimes_of_rows,
                 regime_order, cluster_of_col, n_clusters, axis):
        self.axis = int(axis)
        self.roles = list(roles)
        self.n_rows, self.n_cols = T.shape
        self.regime_order = tuple(regime_order)
        self.n_clusters = int(n_clusters)
        # (regime index, cluster index, row positions, col positions, block)
        self.blocks: list[tuple[int, int, np.ndarray, np.ndarray, _Block]] = []

        for ri, r in enumerate(self.regime_order):
            rows = np.flatnonzero(regimes_of_rows == r)
            if not len(rows):
                continue
            for ci in range(self.n_clusters):
                cols = np.flatnonzero(cluster_of_col == ci)
                if not len(cols):
                    continue
                sl = np.ix_(rows, cols)
                mems = [np.ascontiguousarray(members[k][sl]) for k in self.roles]
                blk = _Block(
                    np.ascontiguousarray(T[sl]),
                    np.ascontiguousarray(B[sl]),
                    mems,
                    self.axis,
                    rows,
                )
                self.blocks.append((ri, ci, rows, cols, blk))
                del mems

        self.outlen = self.n_cols if self.axis == 0 else self.n_rows
        self._by_key = {(ri, ci): (rows, cols, blk)
                        for ri, ci, rows, cols, blk in self.blocks}

        # The first three co-moments (n, sum T, sum T^2) do not depend on the
        # weights at all, so they are accumulated once here and never touched
        # again -- which also means no incremental update can drift them.
        self._const = [np.zeros(self.outlen, dtype=np.float64) for _ in range(3)]
        for ri, ci, rows, cols, blk in self.blocks:
            idx = cols if self.axis == 0 else rows
            for a, v in zip(self._const, (blk.n, blk.sT, blk.sTT)):
                a[idx] += v

        self._var: list[np.ndarray] | None = None   # [sP, sTP, sPP] accumulators
        self._mo: dict[tuple[int, int], tuple] = {}  # per-block contributions
        self._W: np.ndarray | None = None
        self._n_incremental = 0
        self.n_full_recomputes = 0
        self.n_block_updates = 0

    # ------------------------------------------------------------------
    #: Incremental updates are exact in float64 but are refreshed periodically
    #: anyway, so accumulated round-off can never grow without bound over a long
    #: coordinate-ascent run.
    REFRESH_EVERY = 2000

    def _block_moments(self, ri: int, ci: int, W: np.ndarray) -> tuple:
        rows, cols, blk = self._by_key[(ri, ci)]
        e = np.empty(len(self.roles) + 1, dtype=np.float64)
        e[0] = 1.0
        e[1:] = W[ri, :, ci]
        _, _, _, sP, sTP, sPP = blk.moments(e)
        self.n_block_updates += 1
        return (sP, sTP, sPP)

    def _full(self, W: np.ndarray) -> None:
        self._var = [np.zeros(self.outlen, dtype=np.float64) for _ in range(3)]
        self._mo = {}
        for ri, ci, rows, cols, blk in self.blocks:
            mo = self._block_moments(ri, ci, W)
            self._mo[(ri, ci)] = mo
            idx = cols if self.axis == 0 else rows
            for a, v in zip(self._var, mo):
                a[idx] += v
        self._W = W.copy()
        self._n_incremental = 0
        self.n_full_recomputes += 1

    def _accumulate(self, W: np.ndarray):
        """Co-moments on the reduction axis, updating only what changed.

        Coordinate ascent perturbs one ``(regime, role, cluster)`` entry at a
        time, and an entry ``(ri, ki, ci)`` enters exactly one block -- the
        ``(ri, ci)`` one. Recomputing all ``n_regimes x n_clusters`` blocks for
        each candidate would therefore do ~30x more arithmetic than the change
        requires. Here the affected blocks are subtracted out and re-added, which
        is what turns the cluster search from infeasible into routine (measured:
        ~52 ms -> a few ms per evaluation at 8 clusters).
        """
        W = np.asarray(W, dtype=np.float64)
        if self._var is None or self._W is None:
            self._full(W)
        elif not np.array_equal(W, self._W):
            # Which (regime, cluster) blocks does this candidate actually touch?
            d = np.any(W != self._W, axis=1)          # (n_regimes, n_clusters)
            keys = [(int(ri), int(ci)) for ri, ci in zip(*np.nonzero(d))
                    if (int(ri), int(ci)) in self._by_key]
            if self._n_incremental >= self.REFRESH_EVERY or len(keys) > len(self.blocks) // 3:
                self._full(W)
            else:
                for key in keys:
                    rows, cols, _ = self._by_key[key]
                    idx = cols if self.axis == 0 else rows
                    old = self._mo[key]
                    new = self._block_moments(key[0], key[1], W)
                    for a, o, nv in zip(self._var, old, new):
                        a[idx] += nv - o
                    self._mo[key] = new
                self._W = W.copy()
                self._n_incremental += 1
        return self._const + self._var

    def refresh(self) -> None:
        """Force a full recompute; used to bound round-off in long searches."""
        if self._W is not None:
            self._full(self._W)

    def _pooled(self, W: np.ndarray):
        """Pooled co-moments: the accumulated arrays summed to scalars."""
        return [float(a.sum()) for a in self._accumulate(W)]

    # -- metrics (identical formulae to harness / step4_fastscore) --------
    @staticmethod
    def _pcc(n, sT, sTT, sP, sTP, sPP):
        with np.errstate(invalid="ignore", divide="ignore"):
            mT, mP = sT / n, sP / n
            vT = sTT / n - mT * mT
            vP = sPP / n - mP * mP
            cov = sTP / n - mT * mP
            r = cov / np.sqrt(vT * vP)
        bad = (n < MIN_N) | ~(vT > 0) | ~(vP > 0)
        return np.clip(np.where(bad, np.nan, r), -1.0, 1.0)

    @staticmethod
    def _r2(n, sT, sTT, sP, sTP, sPP):
        with np.errstate(invalid="ignore", divide="ignore"):
            mT = sT / n
            vT = sTT / n - mT * mT
            ss_tot = vT * n
            ss_res = sTT - 2.0 * sTP + sPP
            r2 = 1.0 - ss_res / ss_tot
        bad = (n < MIN_N) | ~(ss_tot > 0)
        return np.where(bad, np.nan, r2)

    def pcc_pooled(self, W):
        m = [np.atleast_1d(np.float64(x)) for x in self._pooled(W)]
        return float(self._pcc(*m)[0])

    def r2_pooled(self, W):
        m = [np.atleast_1d(np.float64(x)) for x in self._pooled(W)]
        return float(self._r2(*m)[0])

    def pcc_mean(self, W):
        return _mean_finite(self._pcc(*self._accumulate(W)))

    def r2_mean(self, W):
        return _mean_finite(self._r2(*self._accumulate(W)))


class ClusterFastScorer:
    """Exact fast evaluator for modules 1-3 under per-cluster blend weights.

    Parameters
    ----------
    coh : dict
        Cohort dict with keys ``Y``, ``D``, ``C_h``, ``y_fb``, ``mu_ctx``,
        ``mu_drug``, ``members``, ``regimes``, ``meta_eval`` -- the same contract
        :mod:`step4_fastscore` uses.
    roles : list of str
        Member roles, defining the second axis of ``W``.
    regime_order : tuple of str
        Row-partition labels, defining the first axis of ``W``.
    cluster_of_col : ndarray of int
        Cluster index per protein, defining the third axis of ``W``. Must be a
        partition: every protein in exactly one cluster.
    n_clusters : int
        Number of clusters. ``n_clusters = 1`` reduces exactly to the Step-4
        scalar-weight scorer, which is what makes the scalar control comparable.
    """

    def __init__(self, coh: dict, roles: list[str], regime_order: tuple[str, ...],
                 cluster_of_col: np.ndarray, n_clusters: int, verbose: bool = True):
        import harness as H

        self.roles = list(roles)
        self.regime_order = tuple(regime_order)
        self.n_clusters = int(n_clusters)
        self.cluster_of_col = np.asarray(cluster_of_col, dtype=np.int64)
        self.spec = H.SCORE_SPEC
        self.mw = dict(H.MODULE_WEIGHTS)
        self.sw = {k: dict(v) for k, v in H.SUBMETRIC_WEIGHTS.items()}

        if self.cluster_of_col.min() < 0 or self.cluster_of_col.max() >= self.n_clusters:
            raise ValueError("cluster_of_col carries an index outside [0, n_clusters)")

        regs = np.asarray(coh["regimes"], dtype=object)
        mem = {k: coh["members"][k] for k in self.roles}
        D, Y = coh["D"], coh["Y"]
        C_h, y_fb = coh["C_h"], coh["y_fb"]
        mu_ctx, mu_drug = coh["mu_ctx"], coh["mu_drug"]
        split = coh["meta_eval"]["split_final"].to_numpy()

        base_abs = np.where(np.isfinite(C_h), C_h, y_fb).astype("float32")
        zero = np.zeros_like(base_abs)
        allrows = np.ones(len(regs), dtype=bool)
        self.tasks: dict[str, _CTask] = {}

        def add(name, T, B, mask, axis):
            if not mask.any():
                return
            self.tasks[name] = _CTask(
                T[mask], B[mask], {k: v[mask] for k, v in mem.items()}, self.roles,
                regs[mask], self.regime_order, self.cluster_of_col,
                self.n_clusters, axis,
            )

        if verbose:
            print(f"  [cfast] precomputing co-moments over "
                  f"{len(self.regime_order)} regimes x {self.n_clusters} clusters ...",
                  flush=True)

        add("m1_p", Y, base_abs, allrows, 0)
        add("m2_s", D, zero, allrows, 1)
        add("m2_p", D, zero, allrows, 0)

        for mod in ("m3_s1_chem", "m3_s2_strain", "m3_s3_both", "m3_time"):
            lbl = {
                "m3_s1_chem": "val_chem_only",
                "m3_s2_strain": "val_strain_only",
                "m3_s3_both": "val_both",
                "m3_time": "val_time",
            }[mod]
            mask = split == lbl
            if not mask.any():
                continue
            mc = np.where(np.isfinite(mu_ctx), mu_ctx, np.nan).astype("float64")
            md = np.where(np.isfinite(mu_drug), mu_drug, np.nan).astype("float64")
            if mod == "m3_s1_chem":
                add(f"{mod}_rc_s", D - mc, -mc, mask, 1)
                add(f"{mod}_rc_p", D - mc, -mc, mask, 0)
            elif mod == "m3_s2_strain":
                add(f"{mod}_rd_s", D - md, -md, mask, 1)
                add(f"{mod}_rd_p", D - md, -md, mask, 0)
            else:
                add(f"{mod}_raw_s", D, zero, mask, 1)
                add(f"{mod}_rc_s", D - mc, -mc, mask, 1)
                add(f"{mod}_rd_s", D - md, -md, mask, 1)
        if verbose:
            nb = sum(len(t.blocks) for t in self.tasks.values())
            print(f"  [cfast] {len(self.tasks)} tasks, {nb} regime x cluster blocks",
                  flush=True)

    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        """Weight-tensor shape ``(n_regimes, n_roles, n_clusters)``."""
        return (len(self.regime_order), len(self.roles), self.n_clusters)

    def module_scores(self, W: np.ndarray) -> dict[str, float]:
        """Module scores for modules 1-3 (Module 4 is not computed here)."""
        W = np.asarray(W, dtype=np.float64)
        if W.shape != self.shape:
            raise ValueError(f"W has shape {W.shape}, expected {self.shape}")
        t = self.tasks
        out: dict[str, float] = {}

        sw = self.sw["m1_abundance"]
        out["m1_abundance"] = (
            sw["pcc_pooled"] * _clip01(t["m1_p"].pcc_pooled(W))
            + sw["pcc_per_protein_mean"] * _clip01(t["m1_p"].pcc_mean(W))
            + sw["r2_pooled"] * _clip01(t["m1_p"].r2_pooled(W))
            + sw["r2_per_protein_mean"] * _clip01(t["m1_p"].r2_mean(W))
        )

        sw = self.sw["m2_fold_change"]
        out["m2_fold_change"] = (
            sw["pcc_per_sample_mean"] * _clip01(t["m2_s"].pcc_mean(W))
            + sw["pcc_pooled"] * _clip01(t["m2_p"].pcc_pooled(W))
        )

        for mod, key in (("m3_s1_chem", "rc"), ("m3_s2_strain", "rd")):
            if f"{mod}_{key}_s" not in t:
                out[mod] = 0.0
                continue
            sw = self.sw[mod]
            ps = [k for k in sw if k.endswith("per_sample_mean")][0]
            pp = [k for k in sw if k.endswith("pooled")][0]
            out[mod] = (
                sw[ps] * _clip01(t[f"{mod}_{key}_s"].pcc_mean(W))
                + sw[pp] * _clip01(t[f"{mod}_{key}_p"].pcc_pooled(W))
            )

        for mod in ("m3_s3_both", "m3_time"):
            if f"{mod}_raw_s" not in t:
                out[mod] = 0.0
                continue
            sw = self.sw[mod]
            out[mod] = (
                sw["pcc_per_sample_mean"] * _clip01(t[f"{mod}_raw_s"].pcc_mean(W))
                + sw["resid_ctx_pcc_per_sample_mean"] * _clip01(t[f"{mod}_rc_s"].pcc_mean(W))
                + sw["resid_drug_pcc_per_sample_mean"] * _clip01(t[f"{mod}_rd_s"].pcc_mean(W))
            )
        return out

    def total(self, W: np.ndarray, m4: float = 0.0) -> float:
        """Weighted total with Module 4 supplied as a constant."""
        ms = self.module_scores(W)
        return sum(self.mw[k] * v for k, v in ms.items()) + self.mw["m4_dep"] * m4

    # ------------------------------------------------------------------
    def validate(self, exact_score_fn, n_trials: int = 5, tol: float = 1e-6,
                 seed: int = 0) -> dict:
        """Assert agreement with the real harness on cluster-weight tensors.

        The trial battery is deliberately adversarial about the *cluster* axis,
        because a bug in the scatter-accumulation would cancel out under any
        cluster-constant weight tensor and hide behind a Step-4-style validation:

        0. all-zero (the pure control anchor)
        1. cluster-constant (must reproduce the Step-4 scalar scorer exactly)
        2. cluster-varying, one cluster zeroed
        3. fully random per (regime, role, cluster)
        4. adversarial: alternate 0 and 1.2 across clusters, so neighbouring
           protein columns carry very different weights

        A mismatch means the fast objective is not the harness objective, so the
        optimiser must not run on it -- this raises.
        """
        rng = np.random.default_rng(seed)
        nR, nK, nC = self.shape
        trials = []
        trials.append(("all_zero", np.zeros((nR, nK, nC))))
        w1 = rng.uniform(0.1, 0.6, size=(nR, nK, 1))
        trials.append(("cluster_constant", np.repeat(w1, nC, axis=2)))
        w2 = rng.uniform(0.0, 0.9, size=(nR, nK, nC))
        if nC > 1:
            w2[:, :, 0] = 0.0
        trials.append(("one_cluster_zeroed", w2))
        trials.append(("fully_random", rng.uniform(0.0, 1.0, size=(nR, nK, nC))))
        alt = np.zeros((nR, nK, nC))
        alt[:, :, ::2] = 1.2
        trials.append(("alternating_0_and_1.2", alt))
        trials = trials[: max(n_trials, 1)]

        rows, worst = [], 0.0
        for name, W in trials:
            exact = exact_score_fn(W)
            fast = self.module_scores(W)
            tw = 0.0
            for mod, fv in fast.items():
                ev = float(exact["module_scores"][mod])
                d = abs(fv - ev)
                worst = max(worst, d)
                tw = max(tw, d)
                rows.append({"trial": name, "module": mod, "exact": ev, "fast": fv,
                             "abs_diff": d})
            print(f"  [cfast] validation '{name}': max |fast - exact| over "
                  f"modules 1-3 = {tw:.3e}", flush=True)
        if worst > tol:
            bad = sorted(rows, key=lambda r: -r["abs_diff"])[:5]
            raise AssertionError(
                f"cluster fast scorer disagrees with the harness by {worst:.3e} > {tol:.0e}; "
                f"worst cases: {bad}. Refusing to optimise against a surrogate that is not "
                "the real objective."
            )
        print(f"  [cfast] VALIDATED against the exact harness over "
              f"{len(trials)} cluster-weight tensors (max deviation {worst:.3e} <= {tol:.0e})",
              flush=True)
        return {"max_abs_deviation": float(worst), "n_trials": len(trials), "tol": tol,
                "trial_names": [t[0] for t in trials], "rows": rows}


# ---------------------------------------------------------------------------
def blend_clusters(coh: dict, W: np.ndarray, roles: list[str],
                   regime_order: tuple[str, ...], cluster_of_col: np.ndarray,
                   ) -> np.ndarray:
    """Combine members with non-negative per-(regime, role, cluster) weights.

    This is the reference implementation the fast scorer is validated against, so
    it is written for transparency rather than speed.
    """
    out = np.zeros_like(coh["C_h"])
    regs = np.asarray(coh["regimes"], dtype=object)
    cl = np.asarray(cluster_of_col, dtype=np.int64)
    for ri, regime in enumerate(regime_order):
        rows = np.flatnonzero(regs == regime)
        if not len(rows):
            continue
        acc = np.zeros((len(rows), out.shape[1]), dtype="float64")
        for ki, role in enumerate(roles):
            wvec = np.asarray(W[ri, ki], dtype=np.float64)[cl]  # per-protein weight
            if np.any(wvec != 0.0):
                acc += wvec[None, :] * coh["members"][role][rows]
        out[rows] = acc.astype("float32")
    return out
