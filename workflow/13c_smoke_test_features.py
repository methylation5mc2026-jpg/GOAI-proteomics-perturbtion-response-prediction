"""Fast smoke test of features.py / step3_data.py on a small synthetic problem.

Checks the mechanics (shapes, dtypes, out-of-vocabulary handling, out-of-fold
routing, NaN propagation, row ordering) without touching the 200 MB parquets, so
bugs surface in seconds rather than after a ten-minute build.
"""

from __future__ import annotations

from repo_paths import WORKFLOW_DIR

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(WORKFLOW_DIR))

import features as F  # noqa: E402
from common import CHEM_COL  # noqa: E402

RNG = np.random.default_rng(0)


def make_toy(n: int = 60, p: int = 40) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Toy cohort: 40 train rows, 20 eval rows with a novel strain and chemical."""
    strains = ["S1", "S2", "S3"]
    chems = [f"C{i}" for i in range(5)]
    meta = pd.DataFrame({
        "sample_ID": [f"X{i}" for i in range(n)],
        "data_source": RNG.choice(["B", "C"], n),
        "instrument": RNG.choice(["I1", "I2"], n),
        "Strains": RNG.choice(strains, n),
        "Medium": RNG.choice(["M1", "M2"], n),
        "Temperature": RNG.choice(["30", "37"], n),
        "pert_time": RNG.choice(["15", "30", "60"], n),
        CHEM_COL: RNG.choice(chems, n),
        "Yeast_cell_plate": RNG.choice(["P1", "P2", "P3"], n),
        "protein_well": RNG.choice(["A1", "B7", "H12"], n),
        "pert_id": RNG.choice(["#1", "#2"], n),
        "sample_role": "treatment",
    })
    # Last 20 rows are the eval set; give them an unseen strain and chemical.
    meta.loc[40:, "Strains"] = "S_NEW"
    meta.loc[40:, CHEM_COL] = "C_NEW"
    meta["split_final"] = ["train"] * 40 + ["val"] * 20

    Y = RNG.normal(12, 2, (n, p)).astype("float32")
    Y[RNG.random((n, p)) < 0.2] = np.nan                  # detection dropout
    D = RNG.normal(0, 0.5, (n, p)).astype("float32")
    D[RNG.random((n, p)) < 0.25] = np.nan
    C = Y - D
    return F.add_derived_columns(meta), Y, D, C


def main() -> None:
    meta, Y, D, C = make_toy()
    n, p = D.shape
    fit = (meta["split_final"] == "train").to_numpy()
    enc = F.EncoderSet(meta, Y, D, C, fit, n_folds=3, seed=7)

    # --- 1. shapes / dtypes / column order --------------------------------
    idx = np.arange(n)
    X, ya, yd, rs = enc.build_block(idx)
    assert list(X.columns) == F.FEATURE_NAMES, "column order drifted from FEATURE_NAMES"
    assert len(X) == n * p == len(ya) == len(yd) == len(rs)
    assert all(str(X[c].dtype) == "float32" for c in F.NUM_FEATURES), "numeric dtype not float32"
    assert all(str(X[c].dtype) == "category" for c in F.CAT_FEATURES), "cat dtype not category"
    print(f"  ok: full block {X.shape}, dtypes correct")

    # --- 2. targets and row ordering --------------------------------------
    assert np.array_equal(np.nan_to_num(ya, nan=-9), np.nan_to_num(Y.ravel(), nan=-9)), \
        "y_abs is not row-major aligned with Y"
    assert np.array_equal(np.nan_to_num(yd, nan=-9), np.nan_to_num(D.ravel(), nan=-9)), \
        "y_delta is not row-major aligned with D"
    assert np.array_equal(rs, np.repeat(np.arange(n), p))
    assert np.array_equal(np.nan_to_num(X["c_ctrl"].to_numpy(), nan=-9),
                          np.nan_to_num(C.ravel(), nan=-9)), "c_ctrl misaligned"
    print("  ok: targets and c_ctrl are row-major aligned")

    # --- 3. cell mask -----------------------------------------------------
    mask = np.isfinite(D)
    Xm, yam, ydm, rsm = enc.build_block(idx, mask)
    assert len(Xm) == int(mask.sum()), "cell_mask row count wrong"
    assert np.isfinite(ydm).all(), "masked build emitted a non-finite Delta target"
    assert np.allclose(ydm, D[mask]), "masked target values misaligned"
    print(f"  ok: cell mask keeps {len(Xm)}/{n * p} rows, targets finite")

    # --- 4. out-of-vocabulary categoricals --------------------------------
    ev = np.flatnonzero(~fit)
    Xe, _, _, _ = enc.build_block(ev)
    assert (Xe["Strains"].astype(str) == "__UNSEEN__").all(), "novel strain not mapped to UNSEEN"
    assert (Xe[CHEM_COL].astype(str) == "__UNSEEN__").all(), "novel chemical not mapped to UNSEEN"
    assert Xe["d_strain"].isna().all(), "strain-keyed feature defined for a novel strain"
    assert Xe["d_drug"].isna().all(), "chemical-keyed feature defined for a novel chemical"
    assert Xe["has_drug_prior"].eq(0).all() and Xe["has_strain_prior"].eq(0).all()
    assert Xe["d_batch"].notna().any(), "batch-keyed feature vanished on the eval rows"
    print("  ok: novel entities are out-of-vocabulary; batch features survive")

    # --- 5. out-of-fold routing -------------------------------------------
    tr = np.flatnonzero(fit)
    Xt, _, _, _ = enc.build_block(tr)
    full_vals = []
    for i in tr:
        codes, _ = F._codes_for(meta.iloc[[i]], enc.full.delta["d_drug"].keys,
                                enc.full.delta["d_drug"].vocab)
        full_vals.append(enc.full.delta["d_drug"].rows_for(codes)[0])
    full_vals = np.asarray(full_vals).ravel()
    served = Xt["d_drug"].to_numpy()
    both = np.isfinite(full_vals) & np.isfinite(served)
    assert both.sum() > 0 and not np.allclose(full_vals[both], served[both]), \
        "fit rows were served the full-fit encoding -> out-of-fold routing is inactive"
    print(f"  ok: OOF routing active (mean|served-full| = "
          f"{np.abs(served[both] - full_vals[both]).mean():.4f})")

    # --- 6. audits run and report sane values -----------------------------
    aud = enc.self_contribution_audit(n_samples=20)
    assert aud["d_drug"]["mean_abs_oof_minus_full"] > 0
    cov = enc.coverage_report({"train": fit, "val": ~fit})
    piv = cov.pivot(index="feature", columns="split", values="frac_cells_defined")
    assert piv.loc["d_strain", "val"] == 0.0, "coverage report disagrees with the build"
    print("  ok: self_contribution_audit and coverage_report agree with build_block")

    # --- 7. assemble_training_matrix chunking ------------------------------
    Xa, ya2, yd2 = F.assemble_training_matrix(
        enc, tr, lambda b: np.isfinite(D[b]), chunk=7, label="smoke")
    Xd, yad, ydd, _ = enc.build_block(tr, np.isfinite(D[tr]))
    assert len(Xa) == len(Xd), "chunked assembly changed the row count"
    assert np.allclose(yd2, ydd) and np.allclose(ya2, yad, equal_nan=True), \
        "chunked assembly changed the targets"
    pd.testing.assert_frame_equal(Xa.reset_index(drop=True), Xd.reset_index(drop=True))
    print("  ok: chunked assembly is identical to a single block")

    # --- 8. regime masking makes a train row look like a genuine OOD row ---
    # This is the property the whole regime-specialist design rests on: if a
    # masked train row is NOT indistinguishable from a real novel-entity row,
    # the specialists are trained on the wrong distribution.
    Xf, _, _, _ = enc.build_block(tr)                     # fully-featured train rows
    Xe2, _, _, _ = enc.build_block(ev)                    # real both-novel rows
    for regime, families in F.REGIMES.items():
        Xm = F.apply_regime_mask(Xf, regime)
        assert list(Xm.columns) == F.FEATURE_NAMES, "masking reordered the columns"
        assert len(Xm) == len(Xf), "masking changed the row count"
        for fam, cols in [("chem", F.CHEM_DEPENDENT), ("strain", F.STRAIN_DEPENDENT)]:
            for c in cols:
                if fam in families:
                    if c in F.CAT_FEATURES:
                        assert (Xm[c].astype(str) == "__UNSEEN__").all(), \
                            f"{regime}: categorical {c} not masked"
                    else:
                        assert Xm[c].isna().all(), f"{regime}: numeric {c} not masked"
                elif c not in (F.CHEM_DEPENDENT if fam == "strain" else F.STRAIN_DEPENDENT):
                    assert Xm[c].equals(Xf[c]), f"{regime}: {c} masked but should not be"
        assert Xm["has_drug_prior"].eq(0).all() if "chem" in families else True
        assert Xm["has_strain_prior"].eq(0).all() if "strain" in families else True
    # A real both-novel row and a both_novel-masked train row must agree on
    # exactly which columns are unavailable.
    Xm_both = F.apply_regime_mask(Xf, "both_novel")
    for c in F.CHEM_DEPENDENT + F.STRAIN_DEPENDENT:
        if c in F.CAT_FEATURES:
            assert (Xe2[c].astype(str) == "__UNSEEN__").all() and \
                   (Xm_both[c].astype(str) == "__UNSEEN__").all(), \
                   f"availability of {c} differs between masked and genuine OOD rows"
        else:
            assert Xe2[c].isna().all() and Xm_both[c].isna().all(), \
                f"availability of {c} differs between masked and genuine OOD rows"
    print("  ok: regime masks reproduce genuine OOD feature availability exactly")

    # --- 9. regime routing ------------------------------------------------
    r = F.regime_for_rows(np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]))
    assert list(r) == ["full", "chem_novel", "strain_novel", "both_novel"], \
        f"regime routing table is wrong: {list(r)}"
    print("  ok: regime routing maps novelty flags to the four regimes")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
