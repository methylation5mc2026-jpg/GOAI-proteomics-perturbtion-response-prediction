"""Shared Step-3 data loading: anchors, masks and the inner tuning split.

Three jobs, all of which must be identical across ``14_``/``15_``/``16_`` or the
models and the scores would disagree:

1. **Load the evaluation tensors** through ``validation_splits.load_eval_data``,
   so ``Delta_true`` and the harness anchor are byte-identical to Step 2's.

2. **Re-derive the matched control anchor independently.**  The harness recovers
   the anchor as ``C = Y - Delta``, which is exact but inherits ``Delta``'s
   missingness: a cell where the treated sample was below detection has no
   anchor even when the control measured it perfectly well.  Since ``C`` is a
   *feature* here (the competition defines ``Delta_pred = y_hat - y_control``, so
   the control profile is known to the participant), that limitation would
   needlessly blank the single strongest feature.  :func:`derive_control_anchor`
   therefore rebuilds the anchor straight from the control samples using Step 1's
   frozen matching rule, and :func:`check_anchor_identity` asserts it agrees with
   ``Y - Delta`` wherever both are defined.  Rebuilding it from the controls also
   removes the awkwardness of a feature computed from the target matrix.

3. **Define the inner hyper-parameter-selection split.**  Tuning on the reported
   ``val_*`` splits would make the headline score a tuned-on number.  Instead a
   chemical-holdout is carved out of ``train`` alone: ``INNER_HOLDOUT_FRAC`` of
   the train chemicals become ``inner_dev``, which mimics the S1 (novel
   chemical) axis that carries the most weight.  Every encoder for the inner loop
   is fitted on ``inner_fit`` only.  Selection therefore never sees a ``val_*``
   row, and the reported scores stay out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import validation_splits as VS
from common import (CHEM_COL, CONTEXT_LEVELS, CONTROL_CHEMS, DATA, ID_COL,
                    QC_CHEMS, SEED, WORKFLOW)
from features import add_derived_columns

__all__ = [
    "INNER_HOLDOUT_FRAC",
    "load_train_val",
    "load_test",
    "derive_control_anchor",
    "check_anchor_identity",
    "inner_split",
]

#: Fraction of *train* chemicals held out as the inner tuning dev set.
INNER_HOLDOUT_FRAC = 0.25


# ---------------------------------------------------------------------------
# Control anchor
# ---------------------------------------------------------------------------
def _control_index(meta_pool: pd.DataFrame, mat_pool: np.ndarray) -> dict[str, dict[tuple, np.ndarray]]:
    """Median control profile per context key, per level of the frozen hierarchy.

    Mirrors ``07_delta_matrix.build_control_index`` with the frozen pooled-vehicle
    policy (DMSO and Water pooled; QC injections excluded).
    """
    index: dict[str, dict[tuple, np.ndarray]] = {}
    for lname, keys in CONTEXT_LEVELS:
        gk = list(map(tuple, meta_pool[keys].astype(str).to_numpy())) if keys \
            else [("__global__",)] * len(meta_pool)
        groups: dict[tuple, list[int]] = {}
        for i, k in enumerate(gk):
            groups.setdefault(k, []).append(i)
        d: dict[tuple, np.ndarray] = {}
        for k, idx in groups.items():
            with np.errstate(all="ignore"):
                d[k] = np.nanmedian(mat_pool[idx], axis=0).astype("float32")
        index[lname] = d
    return index


def derive_control_anchor(meta_treat: pd.DataFrame, ctrl_pools: list[tuple[pd.DataFrame, np.ndarray]],
                          label: str = "anchor") -> tuple[np.ndarray, np.ndarray]:
    """Matched control profile per treated sample, via Step 1's frozen hierarchy.

    Parameters
    ----------
    meta_treat : pandas.DataFrame
        Treated-sample metadata; row order defines output row order.
    ctrl_pools : list of (DataFrame, ndarray)
        Control anchor pools, **concatenated into a single pool** before the
        per-context median is taken.  This reproduces Step 1's rule exactly:
        ``07_delta_matrix.py`` stacks the test controls and the train_val
        controls into one anchor pool and takes one median per context group.
        Indexing the pools separately and preferring the first that resolves
        would give a *different* median and therefore a different anchor -- which
        is precisely what :func:`check_anchor_identity` caught when this function
        first did that.
    label : str
        Progress-log label.

    Returns
    -------
    C : numpy.ndarray
        ``(n_treated, n_proteins)`` float32 anchor, NaN where the matched control
        group never detected that protein.
    level_used : numpy.ndarray
        ``(n_treated,)`` object array naming the hierarchy level each sample
        resolved at.
    """
    metas, mats = [], []
    for meta_pool, mat_pool in ctrl_pools:
        keep = meta_pool[CHEM_COL].isin(CONTROL_CHEMS).to_numpy()
        n_qc = int(meta_pool[CHEM_COL].isin(QC_CHEMS).sum())
        print(f"  [{label}] pool: {int(keep.sum())} vehicle controls "
              f"({n_qc} QC injections excluded)", flush=True)
        metas.append(meta_pool.loc[keep])
        mats.append(mat_pool[keep])
    pool_meta = pd.concat(metas, ignore_index=True)
    pool_mat = np.vstack(mats)
    print(f"  [{label}] merged anchor pool: {len(pool_meta)} controls", flush=True)
    index = _control_index(pool_meta, pool_mat)

    n, p = len(meta_treat), pool_mat.shape[1]
    C = np.full((n, p), np.nan, dtype="float32")
    level_used = np.full(n, None, dtype=object)

    keyvals = {lname: (list(map(tuple, meta_treat[keys].astype(str).to_numpy())) if keys
                       else [("__global__",)] * n)
               for lname, keys in CONTEXT_LEVELS}
    for i in range(n):
        for lname, _ in CONTEXT_LEVELS:
            hit = index[lname].get(keyvals[lname][i])
            if hit is not None:
                C[i] = hit
                level_used[i] = lname
                break
        if level_used[i] is None:
            raise RuntimeError(f"{label}: sample {i} unresolved after the global fallback")
        if (i + 1) % 1000 == 0:
            print(f"  [{label}] {i + 1}/{n} anchors resolved ...", flush=True)

    usage = {str(k): int(v) for k, v in pd.Series(level_used).value_counts().items()}
    print(f"  [{label}] defined on {100 * np.isfinite(C).mean():.3f}% of cells; "
          f"level usage {usage}", flush=True)
    return C, level_used


def check_anchor_identity(C_derived: np.ndarray, C_harness: np.ndarray,
                          label: str = "anchor", atol: float = 1e-4) -> dict[str, object]:
    """Assert the re-derived anchor matches ``Y - Delta`` where both are defined.

    A mismatch would mean the feature-side anchor and the harness-side anchor are
    different quantities, which would silently decouple ``Delta_pred`` from
    ``Delta_true``.
    """
    both = np.isfinite(C_derived) & np.isfinite(C_harness)
    dev = np.abs(C_derived[both] - C_harness[both])
    rep = {
        "n_cells_both_defined": int(both.sum()),
        "max_abs_deviation": float(dev.max()) if dev.size else 0.0,
        "mean_abs_deviation": float(dev.mean()) if dev.size else 0.0,
        "frac_defined_derived": round(float(np.isfinite(C_derived).mean()), 6),
        "frac_defined_harness": round(float(np.isfinite(C_harness).mean()), 6),
        "coverage_gain_pp": round(100 * float(np.isfinite(C_derived).mean()
                                              - np.isfinite(C_harness).mean()), 4),
    }
    print(f"  [{label}] identity check: max|dev|={rep['max_abs_deviation']:.2e} over "
          f"{rep['n_cells_both_defined']:,} shared cells; coverage "
          f"{100 * rep['frac_defined_harness']:.2f}% -> {100 * rep['frac_defined_derived']:.2f}% "
          f"(+{rep['coverage_gain_pp']:.2f} pp)", flush=True)
    if dev.size and dev.max() > atol:
        raise AssertionError(
            f"{label}: re-derived anchor deviates from Y - Delta by up to "
            f"{dev.max():.3e} (> {atol}); the two anchors are not the same quantity")
    return rep


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_train_val(verbose: bool = True) -> dict[str, object]:
    """Load train_val treated rows with both anchors and the split masks."""
    meta, Y, D, C_harness, proteins = VS.load_eval_data()
    masks = VS.split_masks(meta)
    meta = add_derived_columns(meta)

    meta_all = pd.read_csv(DATA / "meta_train_val_annotated.csv", dtype=str)
    pr = pd.read_parquet(WORKFLOW / "processed_train_val_proteome.parquet")
    if list(pr[ID_COL].astype(str)) != list(meta_all[ID_COL].astype(str)):
        raise ValueError("train_val proteome is not row-aligned with metadata")
    M_all = pr[proteins].to_numpy(dtype="float32")

    if verbose:
        print("  [anchor] re-deriving the train_val control anchor from control samples ...",
              flush=True)
    C, lvl = derive_control_anchor(meta, [(meta_all, M_all)], "anchor_tv")
    anchor_report = check_anchor_identity(C, C_harness, "anchor_tv")
    anchor_report["level_usage"] = {str(k): int(v) for k, v in
                                    pd.Series(lvl).value_counts().items()}

    # Fill the few cells the re-derivation misses but Y - Delta resolves, so the
    # feature is never *worse* covered than the harness anchor.
    gap = ~np.isfinite(C) & np.isfinite(C_harness)
    if gap.any():
        C[gap] = C_harness[gap]
        anchor_report["cells_backfilled_from_harness_anchor"] = int(gap.sum())
        print(f"  [anchor_tv] backfilled {int(gap.sum()):,} cells from Y - Delta", flush=True)

    return {"meta": meta, "Y": Y, "D": D, "C": C, "C_harness": C_harness,
            "proteins": proteins, "masks": masks, "anchor_report": anchor_report,
            "meta_all": meta_all, "M_all": M_all}


def load_test(proteins: list[str], meta_all_tv: pd.DataFrame, M_all_tv: np.ndarray,
              verbose: bool = True) -> dict[str, object]:
    """Load the independent test treated rows and their matched control anchor.

    The test anchor pool is the test controls first, then the train_val controls
    -- Step 1's frozen rule, motivated by test controls existing almost only for
    the held-out strain.
    """
    dl = pd.read_parquet(WORKFLOW / "processed_delta_matrix_test.parquet")
    meta_cols = [ID_COL, "match_level", "split_final", CHEM_COL]
    if [c for c in dl.columns if c not in meta_cols] != proteins:
        raise ValueError("test delta matrix protein order differs from train_val")

    pr = pd.read_parquet(WORKFLOW / "processed_test_proteome.parquet")
    meta_all = pd.read_csv(DATA / "meta_test_annotated.csv", dtype=str)
    if list(pr[ID_COL].astype(str)) != list(meta_all[ID_COL].astype(str)):
        raise ValueError("test proteome is not row-aligned with metadata")
    M_all = pr[proteins].to_numpy(dtype="float32")

    d_ids = dl[ID_COL].astype(str).to_numpy()
    pos = pd.Series(np.arange(len(meta_all)), index=meta_all[ID_COL].astype(str).to_numpy())
    order = pos.reindex(d_ids).to_numpy()
    if np.isnan(order.astype(float)).any():
        raise ValueError("test delta sample_IDs absent from the test metadata")

    meta = add_derived_columns(meta_all.iloc[order].reset_index(drop=True))
    Y = M_all[order]
    D = dl[proteins].to_numpy(dtype="float32")
    if (meta["sample_role"] != "treatment").any():
        raise ValueError("test delta matrix contains non-treatment rows")

    if verbose:
        print(f"  [test] {len(meta)} treated test samples; "
              f"splits {dict(meta['split_final'].value_counts())}", flush=True)
    C, lvl = derive_control_anchor(meta, [(meta_all, M_all), (meta_all_tv, M_all_tv)],
                                  "anchor_test")
    rep = check_anchor_identity(C, Y - D, "anchor_test")
    rep["level_usage"] = {str(k): int(v) for k, v in pd.Series(lvl).value_counts().items()}
    gap = ~np.isfinite(C) & np.isfinite(Y - D)
    if gap.any():
        C[gap] = (Y - D)[gap]
        rep["cells_backfilled_from_harness_anchor"] = int(gap.sum())
    return {"meta": meta, "Y": Y, "D": D, "C": C, "anchor_report": rep}


# ---------------------------------------------------------------------------
# Inner tuning split
# ---------------------------------------------------------------------------
def inner_split(meta: pd.DataFrame, masks: dict[str, np.ndarray],
                frac: float = INNER_HOLDOUT_FRAC, seed: int = SEED,
                ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Chemical-holdout split *inside* ``train``, for hyper-parameter selection.

    Returns ``(inner_fit_mask, inner_dev_mask, held_out_chemicals)``.  The dev
    chemicals are absent from ``inner_fit`` entirely, so the dev set is a
    novel-chemical (S1-like) generalisation test built without touching any
    ``val_*`` row.
    """
    tr = masks[VS.TRAIN_SPLIT]
    chems = sorted(set(meta.loc[tr, CHEM_COL].astype(str)))
    rng = np.random.default_rng(seed)
    n_hold = max(2, int(round(frac * len(chems))))
    held = sorted(rng.choice(chems, size=n_hold, replace=False).tolist())
    is_held = meta[CHEM_COL].astype(str).isin(held).to_numpy()
    dev = tr & is_held
    fit = tr & ~is_held
    print(f"  [inner] {len(chems)} train chemicals -> {n_hold} held out "
          f"({', '.join(held[:6])}{' ...' if len(held) > 6 else ''})", flush=True)
    print(f"  [inner] inner_fit n={int(fit.sum())}, inner_dev n={int(dev.sum())}", flush=True)
    assert not (set(meta.loc[fit, CHEM_COL].astype(str)) & set(held)), \
        "inner_fit still contains a held-out chemical"
    assert dev.sum() > 0 and fit.sum() > 0, "degenerate inner split"
    return fit, dev, held


#: Fraction of the remaining train samples used as the both-entities-seen dev set.
INNER_RANDOM_FRAC = 0.08


def inner_regime_splits(meta: pd.DataFrame, masks: dict[str, np.ndarray],
                        chem_frac: float = INNER_HOLDOUT_FRAC, seed: int = SEED,
                        ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Four regime-matched dev sets carved out of ``train``, plus the fit mask.

    Each availability regime needs its own selection signal, because the best
    model for "known drug, new strain" is not the best model for "new drug, known
    strain" -- they see disjoint feature families.  Holding out a set of chemicals
    *and* one strain simultaneously partitions the held-out rows into exactly the
    four regimes:

    ==================  ==========================  =========================
    dev set             held-out entity             mirrors
    ==================  ==========================  =========================
    ``chem_novel``      chemical only               S1 / ``val_chem_only``
    ``strain_novel``    strain only                 S2 / ``val_strain_only``
    ``both_novel``      chemical *and* strain       S3 / ``val_both``
    ``full``            neither (random rows)       ``val_time`` / in-domain
    ==================  ==========================  =========================

    Only one strain is held out (``train`` has five), which is the finest
    granularity the data allows on that axis; the resulting selection signal is
    noisier than the chemical one, and that limitation is recorded in ``info``.

    Returns
    -------
    fit_mask : numpy.ndarray
        Rows every inner encoder and candidate model may be fitted on.
    devs : dict
        regime name -> boolean row mask.
    info : dict
        Held-out entities and sizes, for the training report.
    """
    tr = masks[VS.TRAIN_SPLIT]
    rng = np.random.default_rng(seed)

    chems = sorted(set(meta.loc[tr, CHEM_COL].astype(str)))
    n_hold = max(2, int(round(chem_frac * len(chems))))
    held_chems = sorted(rng.choice(chems, size=n_hold, replace=False).tolist())

    strains = sorted(set(meta.loc[tr, "Strains"].astype(str)))
    held_strain = str(rng.choice(strains))

    is_hc = meta[CHEM_COL].astype(str).isin(held_chems).to_numpy()
    is_hs = (meta["Strains"].astype(str) == held_strain).to_numpy()

    devs = {
        "chem_novel": tr & is_hc & ~is_hs,
        "strain_novel": tr & ~is_hc & is_hs,
        "both_novel": tr & is_hc & is_hs,
    }
    rest = tr & ~is_hc & ~is_hs
    rest_idx = np.flatnonzero(rest)
    n_rand = max(30, int(round(INNER_RANDOM_FRAC * len(rest_idx))))
    pick = rng.choice(rest_idx, size=min(n_rand, len(rest_idx)), replace=False)
    full_dev = np.zeros(len(meta), dtype=bool)
    full_dev[pick] = True
    devs["full"] = full_dev
    fit = rest & ~full_dev

    print(f"  [inner] held out {n_hold}/{len(chems)} chemicals and strain "
          f"'{held_strain}' of {strains}", flush=True)
    for k, m in devs.items():
        print(f"  [inner] dev_{k:13s} n={int(m.sum()):5d}", flush=True)
    print(f"  [inner] inner_fit n={int(fit.sum())}", flush=True)

    # The fit set must be clean on both axes, or the dev sets are not OOD.
    assert not (set(meta.loc[fit, CHEM_COL].astype(str)) & set(held_chems)), \
        "inner_fit contains a held-out chemical"
    assert held_strain not in set(meta.loc[fit, "Strains"].astype(str)), \
        "inner_fit contains the held-out strain"
    for k, m in devs.items():
        assert m.sum() > 0, f"dev set {k} is empty"
        assert not (m & fit).any(), f"dev set {k} overlaps inner_fit"
    assert int(fit.sum()) > 500, f"inner_fit too small ({int(fit.sum())} rows)"

    info = {
        "held_out_chemicals": held_chems,
        "held_out_strain": held_strain,
        "n_train_chemicals": len(chems),
        "n_train_strains": len(strains),
        "random_frac_for_full_regime": INNER_RANDOM_FRAC,
        "n_inner_fit": int(fit.sum()),
        "dev_sizes": {k: int(m.sum()) for k, m in devs.items()},
        "limitation": ("train contains only 5 strains, so the novel-strain axis can be probed "
                       "one strain at a time; the strain_novel dev set is a single-strain "
                       "estimate and its selection signal is noisier than the chemical one"),
    }
    return fit, devs, info
