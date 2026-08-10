"""Exact-but-fast re-evaluation of the harness total for linear member blends.

Why
---
One exact :func:`harness.compute_competition_score` call on the validation cohort
costs about 20 s (measured in ``25c_time_harness.py``: 5.7 s for Module 1, 3.5 s
for Module 2, 2.7 s per Module-3 term, 7.1 s for Module 4). Coordinate ascent
over 16 stacking weights needs a few thousand evaluations to converge; at 20 s
each that is over a day. A search truncated to ~45 evaluations would barely move
off its starting point, and reporting its output as "the optimum" would be
wrong.

The key observation
-------------------
Every candidate prediction is an affine function of the weights,
``P = B + sum_k w_k F_k``, and -- crucially -- **the finite-cell mask does not
depend on the weights**, because the member matrices are finite everywhere and
only the truth side (and the frozen baseline ``mu``) carries NaNs. So all seven
masked co-moments the harness needs are *polynomial* in ``w``:

    n      = sum(M)                                   (constant)
    s_T    = sum(M*T)                                  (constant)
    s_TT   = sum(M*T^2)                                (constant)
    s_P    = sum(M*P)     = e . a                      (linear)
    s_TP   = sum(M*T*P)   = e . b                      (linear)
    s_PP   = sum(M*P^2)   = e^T G e                    (quadratic)
    ss_res = s_TT - 2*s_TP + s_PP

with ``e = [1, w_1, ..., w_K]`` and ``a_j = sum(M*F_j)``, ``b_j = sum(M*T*F_j)``,
``G_jl = sum(M*F_j*F_l)`` where ``F_0 = B``. Precomputing ``a``, ``b`` and ``G``
once turns each subsequent evaluation into a handful of small dot products.

Per-regime weights are handled by partitioning rows into regime blocks: a
pseudo-member is a member restricted to one regime, so cross-regime Gram entries
are identically zero and only the within-block Grams need storing.

Scope, stated plainly
---------------------
Modules 1, 2 and 3 (0.95 of the total weight) are reproduced **exactly** -- the
same formulae, the same ``MIN_N`` and zero-variance NaN rules, the same
``clip01``, float64 accumulation throughout. Module 4 (0.05) thresholds
``|Delta| > 1`` and is therefore *not* a polynomial in ``w``; it is held constant
during the search and every shortlisted candidate is then re-scored with the real
harness, so **no reported number ever comes from this module**. It is a search
accelerator, not a scorer.

:meth:`FastScorer.validate` asserts agreement with the real harness on random
weight vectors before the optimiser is allowed to use it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WORKFLOW = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKFLOW))

MIN_N = 3
CHUNK = 512


def _clip01(x: float) -> float:
    """Harness-identical mapping of a raw metric onto [0, 1]."""
    if x is None or not np.isfinite(x):
        return 0.0
    return float(min(max(x, 0.0), 1.0))


def _mean_finite(v: np.ndarray) -> float:
    """Mean over defined slices only, as ``harness._summarise`` does."""
    f = v[np.isfinite(v)]
    return float(f.mean()) if f.size else float("nan")


class _Block:
    """Precomputed co-moments for one regime block of one task.

    Attributes hold, for the requested reduction axis, the constant terms
    (``n``, ``sT``, ``sTT``) and the weight-dependent coefficient arrays
    (``a``, ``b``, ``G``) described in the module docstring.
    """

    __slots__ = ("n", "sT", "sTT", "a", "b", "G", "rows")

    def __init__(self, T, B, mems, axis, rows):
        """Reduce one block.

        Parameters
        ----------
        T : ndarray
            Truth matrix for the block (may contain NaN).
        B : ndarray
            Fixed offset of the prediction (the pseudo-member with coefficient 1).
        mems : list of ndarray
            The weight-carrying member matrices for this block.
        axis : {0, 1}
            ``0`` reduces over rows (one value per protein); ``1`` reduces over
            proteins (one value per row).
        """
        self.rows = rows
        nr, p = T.shape
        K = len(mems)
        outlen = p if axis == 0 else nr

        self.n = np.zeros(outlen, dtype=np.float64)
        self.sT = np.zeros(outlen, dtype=np.float64)
        self.sTT = np.zeros(outlen, dtype=np.float64)
        self.a = np.zeros((K + 1, outlen), dtype=np.float64)
        self.b = np.zeros((K + 1, outlen), dtype=np.float64)
        self.G = np.zeros((K + 1, K + 1, outlen), dtype=np.float64)

        red = 0 if axis == 0 else 1
        for s in range(0, p, CHUNK):
            e = min(s + CHUNK, p)
            t = np.asarray(T[:, s:e], dtype=np.float64)
            bb = np.asarray(B[:, s:e], dtype=np.float64)
            m = np.isfinite(t) & np.isfinite(bb)
            t = np.where(m, t, 0.0)
            mf = m.astype(np.float64)
            F = [np.where(m, bb, 0.0)] + [
                np.where(m, np.asarray(mm[:, s:e], dtype=np.float64), 0.0) for mm in mems
            ]

            def put(dst, val, s=s, e=e):
                if axis == 0:
                    dst[s:e] += val
                else:
                    dst += val

            put(self.n, mf.sum(axis=red))
            put(self.sT, t.sum(axis=red))
            put(self.sTT, (t * t).sum(axis=red))
            for j in range(K + 1):
                put(self.a[j], F[j].sum(axis=red))
                put(self.b[j], (t * F[j]).sum(axis=red))
                for l in range(j, K + 1):
                    g = (F[j] * F[l]).sum(axis=red)
                    put(self.G[j, l], g)
                    if l != j:
                        put(self.G[l, j], g)
            del F, t, bb, m, mf

    def moments(self, e: np.ndarray):
        """Co-moments for coefficient vector ``e`` (with ``e[0] == 1``)."""
        sP = e @ self.a
        sTP = e @ self.b
        sPP = np.einsum("j,jlk,l->k", e, self.G, e, optimize=True)
        return self.n, self.sT, self.sTT, sP, sTP, sPP


class _Task:
    """One (truth, offset, row-subset, axis) configuration across regime blocks."""

    def __init__(self, T, B, members_by_regime, regimes_of_rows, regime_order, axis):
        self.axis = axis
        self.blocks: list[tuple[str, _Block]] = []
        for r in regime_order:
            rows = np.flatnonzero(regimes_of_rows == r)
            if not len(rows):
                continue
            mems = [members_by_regime[k][rows] for k in members_by_regime]
            self.blocks.append(
                (r, _Block(T[rows], B[rows], mems, axis, rows))
            )
        self.keys = list(members_by_regime)

    def _agg(self, W_by_regime):
        """Pooled (axis=None) co-moments: sum the block contributions."""
        n = sT = sTT = sP = sTP = sPP = 0.0
        for r, blk in self.blocks:
            e = np.concatenate([[1.0], W_by_regime[r]])
            bn, bsT, bsTT, bsP, bsTP, bsPP = blk.moments(e)
            n += bn.sum()
            sT += bsT.sum()
            sTT += bsTT.sum()
            sP += bsP.sum()
            sTP += bsTP.sum()
            sPP += bsPP.sum()
        return n, sT, sTT, sP, sTP, sPP

    def _per_slice(self, W_by_regime):
        """Per-slice co-moments, concatenated over blocks (axis=1) or summed (axis=0)."""
        if self.axis == 1:
            outs = []
            for r, blk in self.blocks:
                e = np.concatenate([[1.0], W_by_regime[r]])
                outs.append(blk.moments(e))
            return [np.concatenate([o[i] for o in outs]) for i in range(6)]
        acc = None
        for r, blk in self.blocks:
            e = np.concatenate([[1.0], W_by_regime[r]])
            mo = blk.moments(e)
            acc = list(mo) if acc is None else [a + b for a, b in zip(acc, mo)]
        return acc

    # -- metrics ---------------------------------------------------------
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
        n, sT, sTT, sP, sTP, sPP = self._agg(W)
        return float(self._pcc(*[np.atleast_1d(np.float64(x)) for x in
                                (n, sT, sTT, sP, sTP, sPP)])[0])

    def r2_pooled(self, W):
        n, sT, sTT, sP, sTP, sPP = self._agg(W)
        return float(self._r2(*[np.atleast_1d(np.float64(x)) for x in
                               (n, sT, sTT, sP, sTP, sPP)])[0])

    def pcc_mean(self, W):
        return _mean_finite(self._pcc(*self._per_slice(W)))

    def r2_mean(self, W):
        return _mean_finite(self._r2(*self._per_slice(W)))


class FastScorer:
    """Exact fast evaluator for modules 1-3; module 4 supplied as a constant."""

    def __init__(self, coh: dict, roles: list[str], regime_order: tuple[str, ...],
                 verbose: bool = True):
        import harness as H

        self.roles = list(roles)
        self.regime_order = tuple(regime_order)
        self.spec = H.SCORE_SPEC
        self.mw = dict(H.MODULE_WEIGHTS)
        self.sw = {k: dict(v) for k, v in H.SUBMETRIC_WEIGHTS.items()}

        regs = np.asarray(coh["regimes"], dtype=object)
        mem = {k: coh["members"][k] for k in self.roles}
        D, Y = coh["D"], coh["Y"]
        C_h, y_fb = coh["C_h"], coh["y_fb"]
        mu_ctx, mu_drug = coh["mu_ctx"], coh["mu_drug"]
        split = coh["meta_eval"]["split_final"].to_numpy()

        # Abundance offset: reconstruct() uses C where finite, else the fallback.
        base_abs = np.where(np.isfinite(C_h), C_h, y_fb).astype("float32")
        zero = np.zeros_like(base_abs)

        def sub(mask):
            return {k: v[mask] for k, v in mem.items()}

        allrows = np.ones(len(regs), dtype=bool)
        self.tasks: dict[str, _Task] = {}

        def add(name, T, B, mask, axis):
            if not mask.any():
                return
            self.tasks[name] = _Task(
                T[mask], B[mask], sub(mask), regs[mask], self.regime_order, axis
            )

        if verbose:
            print("  [fast] precomputing co-moments ...", flush=True)

        # Module 1 -- absolute abundance, all rows
        add("m1_p", Y, base_abs, allrows, 0)   # per-protein and pooled
        # Module 2 -- fold change, all rows
        add("m2_s", D, zero, allrows, 1)       # per-sample
        add("m2_p", D, zero, allrows, 0)       # pooled (summed over proteins)

        # Module 3 -- residual PCC on the OOD splits
        for mod, lbl in (
            ("m3_s1_chem", "val_chem_only"),
            ("m3_s2_strain", "val_strain_only"),
            ("m3_s3_both", "val_both"),
            ("m3_time", "val_time"),
        ):
            mask = split == lbl
            if not mask.any():
                continue
            mc = np.where(np.isfinite(mu_ctx), mu_ctx, np.nan).astype("float64")
            md = np.where(np.isfinite(mu_drug), mu_drug, np.nan).astype("float64")
            if mod in ("m3_s1_chem",):
                add(f"{mod}_rc_s", D - mc, -mc, mask, 1)
                add(f"{mod}_rc_p", D - mc, -mc, mask, 0)
            elif mod in ("m3_s2_strain",):
                add(f"{mod}_rd_s", D - md, -md, mask, 1)
                add(f"{mod}_rd_p", D - md, -md, mask, 0)
            else:
                add(f"{mod}_raw_s", D, zero, mask, 1)
                add(f"{mod}_rc_s", D - mc, -mc, mask, 1)
                add(f"{mod}_rd_s", D - md, -md, mask, 1)
        if verbose:
            print(f"  [fast] {len(self.tasks)} tasks precomputed", flush=True)

    # ------------------------------------------------------------------
    def _W(self, W: np.ndarray) -> dict:
        return {r: np.asarray(W[i], dtype=np.float64)
                for i, r in enumerate(self.regime_order)}

    def module_scores(self, W: np.ndarray) -> dict[str, float]:
        """Module scores for modules 1-3 (Module 4 is not computed here)."""
        Wd = self._W(W)
        t = self.tasks
        out: dict[str, float] = {}

        sw = self.sw["m1_abundance"]
        out["m1_abundance"] = (
            sw["pcc_pooled"] * _clip01(t["m1_p"].pcc_pooled(Wd))
            + sw["pcc_per_protein_mean"] * _clip01(t["m1_p"].pcc_mean(Wd))
            + sw["r2_pooled"] * _clip01(t["m1_p"].r2_pooled(Wd))
            + sw["r2_per_protein_mean"] * _clip01(t["m1_p"].r2_mean(Wd))
        )

        sw = self.sw["m2_fold_change"]
        out["m2_fold_change"] = (
            sw["pcc_per_sample_mean"] * _clip01(t["m2_s"].pcc_mean(Wd))
            + sw["pcc_pooled"] * _clip01(t["m2_p"].pcc_pooled(Wd))
        )

        for mod, key in (("m3_s1_chem", "rc"), ("m3_s2_strain", "rd")):
            if f"{mod}_{key}_s" not in t:
                out[mod] = 0.0
                continue
            sw = self.sw[mod]
            ps = [k for k in sw if k.endswith("per_sample_mean")][0]
            pp = [k for k in sw if k.endswith("pooled")][0]
            out[mod] = (
                sw[ps] * _clip01(t[f"{mod}_{key}_s"].pcc_mean(Wd))
                + sw[pp] * _clip01(t[f"{mod}_{key}_p"].pcc_pooled(Wd))
            )

        for mod in ("m3_s3_both", "m3_time"):
            if f"{mod}_raw_s" not in t:
                out[mod] = 0.0
                continue
            sw = self.sw[mod]
            out[mod] = (
                sw["pcc_per_sample_mean"] * _clip01(t[f"{mod}_raw_s"].pcc_mean(Wd))
                + sw["resid_ctx_pcc_per_sample_mean"]
                * _clip01(t[f"{mod}_rc_s"].pcc_mean(Wd))
                + sw["resid_drug_pcc_per_sample_mean"]
                * _clip01(t[f"{mod}_rd_s"].pcc_mean(Wd))
            )
        return out

    def total(self, W: np.ndarray, m4: float = 0.0) -> float:
        """Weighted total with Module 4 supplied as a constant."""
        ms = self.module_scores(W)
        return sum(self.mw[k] * v for k, v in ms.items()) + self.mw["m4_dep"] * m4

    # ------------------------------------------------------------------
    def validate(self, coh: dict, exact_score_fn, n_trials: int = 3,
                 tol: float = 1e-6, seed: int = 0) -> dict:
        """Assert agreement with the real harness on random weight vectors.

        Compares the modules this class reproduces (1, 2 and 3) one by one. A
        mismatch means the fast objective is not the harness objective, so the
        optimiser must not be allowed to run on it -- this raises.
        """
        rng = np.random.default_rng(seed)
        nR, nK = len(self.regime_order), len(self.roles)
        rows = []
        worst = 0.0
        for trial in range(n_trials):
            W = (
                np.zeros((nR, nK))
                if trial == 0
                else (np.full((nR, nK), 0.5) if trial == 1
                      else rng.uniform(0.0, 1.0, size=(nR, nK)))
            )
            exact = exact_score_fn(W)
            fast = self.module_scores(W)
            for mod, fv in fast.items():
                ev = float(exact["module_scores"][mod])
                d = abs(fv - ev)
                worst = max(worst, d)
                rows.append({"trial": trial, "module": mod, "exact": ev,
                             "fast": fv, "abs_diff": d})
            print(f"  [fast] validation trial {trial}: max |fast - exact| over "
                  f"modules 1-3 = "
                  f"{max(r['abs_diff'] for r in rows if r['trial'] == trial):.3e}",
                  flush=True)
        if worst > tol:
            bad = sorted(rows, key=lambda r: -r["abs_diff"])[:5]
            raise AssertionError(
                f"fast scorer disagrees with the harness by {worst:.3e} > {tol:.0e}; "
                f"worst cases: {bad}. Refusing to optimise against a surrogate that is "
                "not the real objective."
            )
        print(f"  [fast] VALIDATED against the exact harness "
              f"(max deviation {worst:.3e} <= {tol:.0e})", flush=True)
        return {"max_abs_deviation": worst, "n_trials": n_trials, "tol": tol,
                "rows": rows}
