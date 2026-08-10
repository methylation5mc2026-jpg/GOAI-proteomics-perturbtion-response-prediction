"""Diagnostic: how expensive is one harness evaluation, and where does it go?

The stacking optimiser's evaluation budget is set by this cost. If a single
score costs tens of seconds, coordinate ascent over 16 weights cannot converge
inside any reasonable wall-clock budget and the search must be restructured
around a cheaper surrogate. Measuring beats guessing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_common as S4  # noqa: E402

WORKFLOW = S4.WORKFLOW


def main() -> None:
    sys.path.insert(0, str(WORKFLOW))
    import harness as H

    d = S4.cache_get("val_lgb_delta")
    C = S4.cache_get("val_C_harness")
    if d is None or C is None:
        raise SystemExit("cached val members not found; run 25b_val_members.py first")
    n, p = d.shape
    print(f"cohort {n} x {p}")

    rng = np.random.default_rng(0)
    Yt = (C + d + rng.normal(0, 0.5, size=d.shape)).astype("float32")
    Dt = (d + rng.normal(0, 0.5, size=d.shape)).astype("float32")
    mu = np.zeros_like(d)

    for name, fn in [
        ("module1_absolute_abundance", lambda: H.module1_absolute_abundance(Yt, C + d)),
        ("module2_fold_change", lambda: H.module2_fold_change(Dt, d)),
        ("module3_residual", lambda: H.module3_residual(Dt, d, mu)),
        ("module4_dep", lambda: H.module4_dep(Dt, d, verbose=False)),
    ]:
        t0 = time.time()
        fn()
        print(f"  {name:32s} {time.time() - t0:6.2f}s")


if __name__ == "__main__":
    main()
