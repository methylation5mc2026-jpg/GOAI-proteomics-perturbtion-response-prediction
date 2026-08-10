"""Local OOD validation splits and train-frozen residual baselines.

The competition evaluates on ``test_chem_only`` / ``test_strain_only`` /
``test_both`` / ``test_time``.  Step 1 derived the matching local splits inside
``train_val`` (``split_final`` in ``data/meta_train_val_annotated.csv``), so this
module turns those labels into an evaluation harness input:

===================  =====  ================================================
``split_final``      n      Generalisation axis
===================  =====  ================================================
``train``            5920   fit / freeze everything here
``val_chem_only``    1065   **S1** unseen chemical, seen strain
``val_strain_only``  1547   **S2** unseen strain, seen chemical
``val_both``          269   **S3** unseen strain *and* unseen chemical
``val_time``          157   unseen time point, seen strain and chemical
===================  =====  ================================================

Three jobs live here:

1. **Load aligned evaluation tensors.**  ``Y`` (log2 abundance), ``Delta_true``
   and the frozen control anchor ``C``, all row-aligned to one metadata frame.
   Only treated samples have a ``Delta``, so the harness operates on the 7,884
   treated rows; controls and QC injections are anchors, not targets.

2. **Prove there is no leakage.**  Every claim the split names make is asserted
   against the data rather than trusted, and the assertions run on import of
   ``main()`` so a silent regression cannot slip through.

3. **Freeze the Module-3 residual baselines.**  ``mu_ctx`` (mean ``Delta`` per
   biological context) and ``mu_drug`` (mean ``Delta`` per chemical) are
   computed from ``train`` rows **only**, per the handbook rule that all
   reference and normalisation statistics must be frozen on training data.
   Groups thinner than :data:`MIN_GROUP_N` fall back to a coarser key so that a
   baseline is never a single noisy observation; the level actually used is
   recorded per sample and reported.

The control anchor is recovered as ``C = Y - Delta_true``.  Step 1's
``09_verify_outputs.py`` re-derived ``Delta`` from the raw log2 matrices with
0.00 deviation over 146,086 checked cells, so this identity is exact wherever
``Delta_true`` is defined -- which is exactly the set of cells the harness
scores.  Cells where the anchor was below detection stay NaN and drop out of
every metric on both the prediction and truth side symmetrically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import CHEM_COL, DATA, ID_COL, WORKFLOW

__all__ = [
    "EVAL_SPLITS",
    "CTX_LEVELS",
    "DRUG_LEVELS",
    "MIN_GROUP_N",
    "load_eval_data",
    "split_masks",
    "check_no_leakage",
    "frozen_delta_baseline",
    "build_residual_baselines",
]

#: Local split names in official evaluation order (S1, S2, S3, Time).
EVAL_SPLITS = ("val_chem_only", "val_strain_only", "val_both", "val_time")

TRAIN_SPLIT = "train"

#: Minimum number of train samples in a group before its mean is trusted as a
#: residual baseline; thinner groups fall through to the next coarser level.
MIN_GROUP_N = 3

#: ``mu_ctx`` fallback hierarchy, finest -> coarsest.  The finest level is the
#: biological context defined in the task spec (Strain x Medium x Temperature x
#: Time).  ``data_source`` is deliberately absent: Delta already cancels the
#: plate/instrument offset that PC1 captures, so conditioning on it here would
#: only thin the groups.
CTX_LEVELS: list[tuple[str, list[str]]] = [
    ("CTX1_strain_medium_temp_time", ["Strains", "Medium", "Temperature", "pert_time"]),
    ("CTX2_strain_medium_temp", ["Strains", "Medium", "Temperature"]),
    ("CTX3_medium_temp_time", ["Medium", "Temperature", "pert_time"]),
    ("CTX4_medium_temp", ["Medium", "Temperature"]),
    ("CTX5_global", []),
]

#: Batch-aware variant of ``CTX_LEVELS``, used only as a *sensitivity* axis --
#: never as the primary metric.  ``10_eval_baselines.py`` shows that scoring the
#: S1/S2 residuals against a batch-blind ``mu_ctx`` lets a purely batch-
#: conditioned group mean earn 0.45 residual PCC with zero chemical knowledge,
#: which collapses to 0.000 once ``data_source`` enters the baseline.  Keeping
#: both definitions makes that confound measurable instead of invisible.
CTX_LEVELS_BATCH: list[tuple[str, list[str]]] = [
    ("BCTX1_batch_strain_medium_temp_time",
     ["data_source", "Strains", "Medium", "Temperature", "pert_time"]),
    ("BCTX2_batch_strain_medium_temp", ["data_source", "Strains", "Medium", "Temperature"]),
    ("BCTX3_batch_medium_temp_time", ["data_source", "Medium", "Temperature", "pert_time"]),
    ("BCTX4_batch_medium_temp", ["data_source", "Medium", "Temperature"]),
    ("BCTX5_batch", ["data_source"]),
    ("BCTX6_global", []),
]

#: ``mu_drug`` fallback hierarchy.  A novel chemical has no train rows by
#: definition, so S1/S3 fall straight to the global mean -- reported, not hidden.
DRUG_LEVELS: list[tuple[str, list[str]]] = [
    ("DRUG1_chemical", [CHEM_COL]),
    ("DRUG2_global", []),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_eval_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the row-aligned evaluation tensors for all treated ``train_val`` rows.

    Returns
    -------
    meta : pandas.DataFrame
        7,884 rows (all treated samples), carrying ``split_final``,
        ``strain_role``, ``chemical_role``, ``match_level`` and the biological
        condition columns.  Row order defines matrix row order.
    Y : numpy.ndarray
        ``(n, p)`` float32 normalised log2 abundance (the prediction target).
    D : numpy.ndarray
        ``(n, p)`` float32 true fold-change ``Y - C``.
    C : numpy.ndarray
        ``(n, p)`` float32 frozen matched-control anchor.
    proteins : list of str
        Protein column names in fixed submission order.
    """
    print("  [splits] loading delta matrix ...", flush=True)
    dl = pd.read_parquet(WORKFLOW / "processed_delta_matrix.parquet")
    meta_cols = [ID_COL, "match_level", "split_final", CHEM_COL]
    proteins = [c for c in dl.columns if c not in meta_cols]
    print(f"  [splits] delta: {dl.shape[0]} treated samples x {len(proteins)} proteins",
          flush=True)

    print("  [splits] loading log2 abundance matrix ...", flush=True)
    pr = pd.read_parquet(WORKFLOW / "processed_train_val_proteome.parquet")
    if [c for c in pr.columns if c != ID_COL] != proteins:
        raise ValueError("protein column order differs between delta and proteome matrices")

    meta_all = pd.read_csv(DATA / "meta_train_val_annotated.csv", dtype=str)
    if list(pr[ID_COL].astype(str)) != list(meta_all[ID_COL].astype(str)):
        raise ValueError("proteome matrix is not row-aligned with annotated metadata")

    # Align metadata + abundance onto the delta row order.
    d_ids = dl[ID_COL].astype(str).to_numpy()
    if len(set(d_ids)) != len(d_ids):
        raise ValueError("duplicate sample_ID in the delta matrix")
    pos = pd.Series(np.arange(len(pr)), index=pr[ID_COL].astype(str).to_numpy())
    missing = [s for s in d_ids if s not in pos.index]
    if missing:
        raise ValueError(f"{len(missing)} delta sample_IDs absent from the proteome matrix")
    order = pos.reindex(d_ids).to_numpy()

    Y = pr[proteins].to_numpy(dtype="float32")[order]
    D = dl[proteins].to_numpy(dtype="float32")
    C = Y - D

    meta = meta_all.iloc[order].reset_index(drop=True)
    meta["match_level"] = dl["match_level"].to_numpy()
    if not (meta[ID_COL].astype(str).to_numpy() == d_ids).all():
        raise ValueError("metadata alignment failed after reindexing")
    if not (meta["split_final"].to_numpy() == dl["split_final"].to_numpy()).all():
        raise ValueError("split_final disagrees between metadata and delta matrix")
    if (meta["sample_role"] != "treatment").any():
        raise ValueError("delta matrix contains non-treatment rows")

    fin_d = float(np.isfinite(D).mean())
    print(f"  [splits] aligned: Y/D/C = {Y.shape}; Delta defined on "
          f"{100 * fin_d:.3f}% of cells", flush=True)
    return meta, Y, D, C, proteins


def split_masks(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean row masks for ``train`` and each evaluation split."""
    sp = meta["split_final"].to_numpy()
    masks = {name: (sp == name) for name in (TRAIN_SPLIT, *EVAL_SPLITS)}
    masks["all_val"] = np.isin(sp, EVAL_SPLITS)
    unknown = set(np.unique(sp)) - set((TRAIN_SPLIT, *EVAL_SPLITS))
    if unknown:
        raise ValueError(f"unexpected split_final values: {sorted(unknown)}")
    for name, m in masks.items():
        print(f"  [splits] {name:18s} n={int(m.sum()):5d}", flush=True)
    return masks


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------
def check_no_leakage(meta: pd.DataFrame, masks: dict[str, np.ndarray]) -> dict[str, object]:
    """Assert every generalisation claim the split names make.

    Verifies sample disjointness, that "novel strain" / "novel chemical" splits
    genuinely contain entities absent from ``train``, that the complementary
    entity *is* present in ``train`` (otherwise the split would not isolate a
    single generalisation axis), and that ``val_time`` is novel in time only.

    Raises
    ------
    AssertionError
        On any leakage or mislabelled split.
    """
    report: dict[str, object] = {}
    tr = masks[TRAIN_SPLIT]
    tr_strain = set(meta.loc[tr, "Strains"])
    tr_chem = set(meta.loc[tr, CHEM_COL])
    tr_ids = set(meta.loc[tr, ID_COL])
    tr_time = set(meta.loc[tr, "pert_time"])

    # 1) Sample-level disjointness across all splits.
    seen: set[str] = set()
    for name in (TRAIN_SPLIT, *EVAL_SPLITS):
        ids = set(meta.loc[masks[name], ID_COL])
        overlap = seen & ids
        assert not overlap, f"{name}: {len(overlap)} sample_IDs also in an earlier split"
        seen |= ids
    assert len(seen) == len(meta), "split masks do not partition the treated rows"
    report["sample_disjoint"] = True

    # 2) Per-split entity semantics.
    #
    # NOTE on val_time: it does *not* hold out an unseen pert_time value.  All
    # six time points (15/30/60/90/120/240 min) appear in train, and every
    # val_time (strain, medium, temperature, chemical) condition is present in
    # train with a mean of 5.30 of the 6 time points observed.  What is held out
    # is the *cell*: 62.6% of val_time rows have a
    # (strain, medium, temperature, chemical, time) combination absent from
    # train, and the remaining 37.4% are the same cell re-measured in a
    # different ``data_source`` batch.  So this split scores within-condition
    # time-grid completion (interpolation) plus measurement reproducibility --
    # not extrapolation beyond the observed time range.  Verified empirically;
    # see ``time_semantics`` in the returned report.
    expect = {
        # split              strain novel?  chemical novel?  condition-cell novel?
        "val_chem_only": (False, True, True),
        "val_strain_only": (True, False, True),
        "val_both": (True, True, True),
        "val_time": (False, False, False),
    }
    cell_keys = ["Strains", "Medium", "Temperature", CHEM_COL, "pert_time"]
    cond_keys = ["Strains", "Medium", "Temperature", CHEM_COL]
    tr_cells = set(map(tuple, meta.loc[tr, cell_keys].astype(str).to_numpy()))
    tr_conds = set(map(tuple, meta.loc[tr, cond_keys].astype(str).to_numpy()))

    for name, (s_novel, c_novel, cell_fully_novel) in expect.items():
        m = masks[name]
        strains = set(meta.loc[m, "Strains"])
        chems = set(meta.loc[m, CHEM_COL])
        s_leak = strains & tr_strain
        c_leak = chems & tr_chem
        if s_novel:
            assert not s_leak, f"{name}: {len(s_leak)} strains leak from train: {sorted(s_leak)[:5]}"
        else:
            assert strains <= tr_strain, f"{name}: strains unexpectedly absent from train"
        if c_novel:
            assert not c_leak, f"{name}: {len(c_leak)} chemicals leak from train: {sorted(c_leak)[:5]}"
        else:
            assert chems <= tr_chem, f"{name}: chemicals unexpectedly absent from train"

        cells = list(map(tuple, meta.loc[m, cell_keys].astype(str).to_numpy()))
        conds = list(map(tuple, meta.loc[m, cond_keys].astype(str).to_numpy()))
        frac_cell_novel = sum(c not in tr_cells for c in cells) / len(cells)
        frac_cond_novel = sum(c not in tr_conds for c in conds) / len(conds)
        if cell_fully_novel:
            # A genuine entity-OOD split cannot re-use any train condition cell.
            assert frac_cell_novel == 1.0, (
                f"{name}: only {100 * frac_cell_novel:.1f}% of condition cells are novel; "
                "an entity-held-out split must be 100% novel")

        report[name] = {
            "n_samples": int(m.sum()),
            "n_strains": len(strains),
            "n_chemicals": len(chems),
            "strain_novel": bool(s_novel),
            "chemical_novel": bool(c_novel),
            "strains_overlapping_train": len(s_leak),
            "chemicals_overlapping_train": len(c_leak),
            "frac_condition_cells_novel": round(frac_cell_novel, 6),
            "frac_conditions_novel_ignoring_time": round(frac_cond_novel, 6),
            "pert_times": sorted(set(meta.loc[m, "pert_time"])),
            "pert_times_unseen_in_train": sorted(set(meta.loc[m, "pert_time"]) - tr_time),
        }
        print(f"  [leak] {name:18s} strains={len(strains):3d} (novel={s_novel}, leak={len(s_leak)}) "
              f"chems={len(chems):3d} (novel={c_novel}, leak={len(c_leak)}) "
              f"novel_cells={100 * frac_cell_novel:5.1f}%", flush=True)

    # 3) Characterise the val_time axis explicitly rather than assuming it.
    vt = masks["val_time"]
    vt_times = set(meta.loc[vt, "pert_time"])
    unseen_times = vt_times - tr_time
    assert not unseen_times, (
        f"val_time contains pert_time values absent from train {sorted(unseen_times)}; "
        "the interpolation-only characterisation below would no longer hold")
    vt_cells = list(map(tuple, meta.loc[vt, cell_keys].astype(str).to_numpy()))
    frac_novel_cells = sum(c not in tr_cells for c in vt_cells) / len(vt_cells)
    assert frac_novel_cells > 0.5, (
        f"only {100 * frac_novel_cells:.1f}% of val_time cells are unseen; the split would "
        "carry little time-completion signal")
    times_per_cond = (meta.loc[tr].groupby(cond_keys)["pert_time"].nunique())
    vt_conds = set(map(tuple, meta.loc[vt, cond_keys].astype(str).to_numpy()))
    cover = [int(times_per_cond.get(c, 0)) for c in vt_conds]
    report["time_semantics"] = {
        "generalisation_axis": "within-condition time-grid completion (interpolation)",
        "unseen_time_values": [],
        "all_time_values": sorted(vt_times),
        "frac_rows_with_novel_condition_cell": round(frac_novel_cells, 6),
        "frac_rows_that_are_batch_replicates_of_a_train_cell": round(1 - frac_novel_cells, 6),
        "n_condition_groups": len(vt_conds),
        "train_time_points_per_condition_mean": round(float(np.mean(cover)), 3),
        "train_time_points_per_condition_min": int(np.min(cover)),
        "note": ("no pert_time value is held out; models may treat pert_time as an "
                 "ordinal/numeric feature and interpolate within the observed range"),
    }
    print(f"  [leak] val_time axis = time-grid completion: "
          f"{100 * frac_novel_cells:.1f}% novel cells, "
          f"{100 * (1 - frac_novel_cells):.1f}% cross-batch replicates; "
          f"train covers {np.mean(cover):.2f}/6 time points per condition", flush=True)

    report["train"] = {
        "n_samples": int(tr.sum()), "n_strains": len(tr_strain),
        "n_chemicals": len(tr_chem), "n_sample_ids": len(tr_ids),
    }
    print("  [leak] all leakage assertions PASSED", flush=True)
    return report


# ---------------------------------------------------------------------------
# Train-frozen residual baselines
# ---------------------------------------------------------------------------
def frozen_delta_baseline(meta: pd.DataFrame, D: np.ndarray, train_mask: np.ndarray,
                          levels: list[tuple[str, list[str]]], label: str,
                          min_group_n: int = MIN_GROUP_N,
                          ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Per-sample mean-``Delta`` baseline fitted on ``train`` rows only.

    For each level of the fallback hierarchy the per-group protein-wise mean of
    the training ``Delta`` is computed with NaN skipped.  Each sample then takes
    the mean from the finest level whose group has at least ``min_group_n``
    training rows.  Cells where a group has no finite training observation stay
    NaN and are excluded downstream.

    Returns
    -------
    mu : numpy.ndarray
        ``(n, p)`` float32 baseline, aligned to ``meta``.
    level_used : numpy.ndarray
        ``(n,)`` object array naming the level each sample resolved to.
    info : dict
        Group counts and level usage, for the diagnostics report.
    """
    n, p = D.shape
    n_train = int(train_mask.sum())
    print(f"  [{label}] fitting on {n_train} train rows across {len(levels)} levels "
          f"(min_group_n={min_group_n}) ...", flush=True)

    Dtr = D[train_mask]
    meta_tr = meta.loc[train_mask]

    mu = np.full((n, p), np.nan, dtype="float32")
    level_used = np.full(n, None, dtype=object)
    unresolved = np.ones(n, dtype=bool)
    info: dict[str, object] = {"levels": [], "min_group_n": min_group_n,
                              "n_train_rows": n_train}

    for lname, keys in levels:
        if not unresolved.any():
            break
        if keys:
            tr_key = pd.MultiIndex.from_frame(meta_tr[keys].astype(str)) if len(keys) > 1 \
                else pd.Index(meta_tr[keys[0]].astype(str))
            all_key = pd.MultiIndex.from_frame(meta[keys].astype(str)) if len(keys) > 1 \
                else pd.Index(meta[keys[0]].astype(str))
        else:
            tr_key = pd.Index(np.repeat("__global__", n_train))
            all_key = pd.Index(np.repeat("__global__", n))

        # Group-wise NaN-skipping mean of the training deltas.
        frame = pd.DataFrame(Dtr, copy=False)
        grp = frame.groupby(tr_key, sort=False)
        means = grp.mean()                      # pandas mean skips NaN
        sizes = grp.size()
        keep = sizes[sizes >= min_group_n].index
        means = means.loc[keep]

        lut = {k: i for i, k in enumerate(means.index)}
        mv = means.to_numpy(dtype="float32")
        tgt = np.array([lut.get(k, -1) for k in all_key], dtype=int)
        take = unresolved & (tgt >= 0)
        if take.any():
            mu[take] = mv[tgt[take]]
            level_used[take] = lname
            unresolved &= ~take

        info["levels"].append({
            "level": lname, "keys": keys,
            "n_groups_total": int(len(sizes)),
            "n_groups_kept": int(len(keep)),
            "n_samples_resolved_here": int(take.sum()),
        })
        print(f"    [{label}] {lname:30s} groups={len(keep):4d}/{len(sizes):4d} "
              f"resolved={int(take.sum()):5d} remaining={int(unresolved.sum()):5d}", flush=True)

    if unresolved.any():
        raise RuntimeError(f"{label}: {int(unresolved.sum())} samples unresolved after the "
                           "global fallback -- hierarchy is misconfigured")

    frac_defined = float(np.isfinite(mu).mean())
    info["level_usage"] = {str(k): int(v) for k, v in
                           pd.Series(level_used).value_counts().items()}
    info["frac_cells_defined"] = round(frac_defined, 6)
    print(f"  [{label}] baseline defined on {100 * frac_defined:.3f}% of cells; "
          f"level usage {info['level_usage']}", flush=True)
    return mu, level_used, info


def build_residual_baselines(meta: pd.DataFrame, D: np.ndarray,
                             masks: dict[str, np.ndarray],
                             ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build both Module-3 residual baselines, frozen on ``train``.

    Returns ``(mu_ctx, mu_drug, diagnostics)``.  A guard confirms that no
    evaluation row contributed to either fit.
    """
    tr = masks[TRAIN_SPLIT]
    mu_ctx, lvl_ctx, info_ctx = frozen_delta_baseline(meta, D, tr, CTX_LEVELS, "mu_ctx")
    mu_drug, lvl_drug, info_drug = frozen_delta_baseline(meta, D, tr, DRUG_LEVELS, "mu_drug")

    # Leakage guard: refit on train only must be invariant to eval-row content.
    D_scrambled = D.copy()
    D_scrambled[~tr] = np.nan
    mu_ctx_check, _, _ = frozen_delta_baseline(meta, D_scrambled, tr, CTX_LEVELS,
                                               "mu_ctx_leakcheck")
    same = np.allclose(np.nan_to_num(mu_ctx, nan=-9e9),
                       np.nan_to_num(mu_ctx_check, nan=-9e9), atol=0, rtol=0)
    assert same, "mu_ctx changed when eval rows were blanked -> evaluation data leaked into the fit"
    print("  [baselines] leakage guard PASSED (baselines depend on train rows only)", flush=True)

    diag: dict[str, object] = {
        "mu_ctx": info_ctx,
        "mu_drug": info_drug,
        "level_usage_by_split": {},
    }
    for name in EVAL_SPLITS:
        m = masks[name]
        diag["level_usage_by_split"][name] = {
            "mu_ctx": {str(k): int(v) for k, v in pd.Series(lvl_ctx[m]).value_counts().items()},
            "mu_drug": {str(k): int(v) for k, v in pd.Series(lvl_drug[m]).value_counts().items()},
        }
    return mu_ctx, mu_drug, diag
