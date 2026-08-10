"""Empirically determine which features vanish on which OOD split.

The Step-3 first pass lost 0.106 of the S2 (novel-strain) module score to the
benchmark while *winning* on the splits where every feature is available.  That
is the signature of a train/serve feature-availability mismatch: the model is
fitted on rows where the strain-keyed features are present (98% of train rows)
and then served rows where they are all NaN, so it is forced down a default
branch that almost no training data shaped.

Fixing it needs an exact inventory of which features are unavailable in which
regime -- assigning that by hand from the column names would be guesswork for the
categoricals (``pert_id``, for instance, is a plate/dose slot code, not a
chemical key, so it may well stay in-vocabulary for a novel chemical).  This
script measures it and writes the inventory used by ``features.REGIMES``.
"""

from __future__ import annotations

from repo_paths import WORKFLOW_DIR

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(WORKFLOW_DIR))

import features as F  # noqa: E402
import step3_data as S3  # noqa: E402
import validation_splits as VS  # noqa: E402
from common import RESULTS, SEED  # noqa: E402


def main() -> None:
    tv = S3.load_train_val(verbose=False)
    meta, Y, D, C, masks = tv["meta"], tv["Y"], tv["D"], tv["C"], tv["masks"]
    train_mask = masks[VS.TRAIN_SPLIT]
    enc = F.EncoderSet(meta, Y, D, C, train_mask, n_folds=F.N_FOLDS, seed=SEED,
                       verbose=False)

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for split in [VS.TRAIN_SPLIT, *VS.EVAL_SPLITS]:
        idx = np.flatnonzero(masks[split])
        take = np.sort(rng.choice(idx, size=min(120, len(idx)), replace=False))
        X, _, _, _ = enc.build_block(take)
        for f in F.NUM_FEATURES:
            rows.append({"split": split, "feature": f, "kind": "numeric",
                         "frac_available": round(float(X[f].notna().mean()), 6)})
        for f in F.CAT_FEATURES:
            in_vocab = (X[f].astype(str) != "__UNSEEN__").mean()
            rows.append({"split": split, "feature": f, "kind": "categorical",
                         "frac_available": round(float(in_vocab), 6)})

    df = pd.DataFrame(rows)
    piv = df.pivot(index="feature", columns="split", values="frac_available")
    piv = piv[[VS.TRAIN_SPLIT, *VS.EVAL_SPLITS]]
    print("\nFeature availability by split (fraction of cells / rows usable):")
    print(piv.round(3).to_string(), flush=True)

    # A feature is "lost" in a regime when it is essentially unavailable there
    # while being available in train.  0.02 tolerates the handful of cells that a
    # thin group leaves undefined.
    tol = 0.02
    avail_train = piv[VS.TRAIN_SPLIT]
    lost_chem = sorted(piv.index[(piv["val_chem_only"] <= tol) & (avail_train > tol)])
    lost_strain = sorted(piv.index[(piv["val_strain_only"] <= tol) & (avail_train > tol)])
    lost_both = sorted(piv.index[(piv["val_both"] <= tol) & (avail_train > tol)])
    lost_time = sorted(piv.index[(piv["val_time"] <= tol) & (avail_train > tol)])

    print(f"\nlost when the chemical is novel  (S1): {lost_chem}")
    print(f"lost when the strain is novel    (S2): {lost_strain}")
    print(f"lost when both are novel         (S3): {lost_both}")
    print(f"lost on the time split                : {lost_time}")

    union = sorted(set(lost_chem) | set(lost_strain))
    print(f"\nS3 loss set == union(S1, S2) loss sets: {sorted(lost_both) == union}")

    out = {
        "tolerance": tol,
        "availability_by_split": {c: piv[c].round(6).to_dict() for c in piv.columns},
        "lost_when_chemical_novel": lost_chem,
        "lost_when_strain_novel": lost_strain,
        "lost_when_both_novel": lost_both,
        "lost_on_time_split": lost_time,
        "s3_is_union_of_s1_and_s2": sorted(lost_both) == union,
        "interpretation": (
            "A model fitted on rows where these features are present and then served rows "
            "where they are absent is evaluated off its training distribution. Training one "
            "specialist per availability regime -- masking exactly these features in the "
            "training matrix -- removes the mismatch while still using every train row."),
    }
    (RESULTS / "gbdt_feature_availability.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote results/gbdt_feature_availability.json", flush=True)


if __name__ == "__main__":
    main()
