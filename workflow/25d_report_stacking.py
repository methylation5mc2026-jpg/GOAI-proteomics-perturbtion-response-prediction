"""Readable summary of the Step-4 stacking result: weights and per-module deltas."""
from __future__ import annotations

from repo_paths import RESULTS_DIR

import json
from pathlib import Path

RESULTS = RESULTS_DIR


def main() -> None:
    s = json.loads((RESULTS / "step4_stacking_weights.json").read_text(encoding="utf-8"))
    roles, mw = s["roles"], s["module_weights"]
    mods = list(mw)

    print("=== frozen per-regime weights (fitted on inner-dev only) ===")
    hdr = f"{'regime':14s} " + " ".join(f"{r:>10s}" for r in roles) + f" {'sum':>8s}"
    print(hdr)
    for reg, w in s["frozen_weights"].items():
        tot = sum(w[r] for r in roles)
        print(f"{reg:14s} " + " ".join(f"{w[r]:10.3f}" for r in roles) + f" {tot:8.3f}")
    print("\n  (sum < 1 == shrinkage toward the control anchor; sum is NOT constrained)")

    print("\n=== per-module scores on the validation cohort ===")
    keys = [
        ("val:stacked_oof_frozen", "STACKED (frozen OOF)"),
        ("val:bench", "bench member"),
        ("val:gbdt_mol", "RDKit LightGBM"),
        ("val:gbdt_tab", "tabular LightGBM"),
        ("val:dl", "MLP-ResNet"),
        ("val:control_anchor", "control anchor"),
    ]
    keys = [(k, lb) for k, lb in keys if k in s["module_scores"]]
    print(f"{'module':16s} {'wt':>5s} " + " ".join(f"{lb[:13]:>14s}" for _, lb in keys))
    for m in mods:
        print(f"{m:16s} {mw[m]:5.2f} "
              + " ".join(f"{s['module_scores'][k].get(m, float('nan')):14.4f}"
                         for k, _ in keys))
    print(f"{'TOTAL':16s} {1.0:5.2f} "
          + " ".join(f"{s['totals'][k]:14.4f}" for k, _ in keys))

    print("\n=== what the stacker gained, per module (weighted, vs bench member) ===")
    st, bm = s["module_scores"]["val:stacked_oof_frozen"], s["module_scores"]["val:bench"]
    net = 0.0
    for m in mods:
        d = mw[m] * (st[m] - bm[m])
        net += d
        tag = "  <-- gain" if d > 0.001 else ("  <-- loss" if d < -0.001 else "")
        print(f"  {m:16s} {d:+.5f}{tag}")
    print(f"  {'NET':16s} {net:+.5f}")

    print("\n=== RDKit effect: molecular vs tabular LightGBM (same config/rows/rounds) ===")
    mol, tab = s["module_scores"]["val:gbdt_mol"], s["module_scores"]["val:gbdt_tab"]
    for m in mods:
        d = mol[m] - tab[m]
        print(f"  {m:16s} {tab[m]:.4f} -> {mol[m]:.4f}   ({d:+.4f})")
    print(f"  TOTAL            {s['totals']['val:gbdt_tab']:.4f} -> "
          f"{s['totals']['val:gbdt_mol']:.4f}   "
          f"({s['totals']['val:gbdt_mol'] - s['totals']['val:gbdt_tab']:+.4f})")

    print("\n=== honesty checks ===")
    print(f"  inner-dev total at frozen weights : {s['inner_dev_total_at_frozen_weights']:.6f}")
    print(f"  val total at frozen weights       : {s['val_total_at_frozen_weights']:.6f}")
    print(f"  inner -> val change               : {s['generalisation_gap_inner_to_val']:+.6f}")
    print(f"  val-TUNED upper bound (optimistic): {s['val_total_val_tuned_OPTIMISTIC']:.6f}")
    print(f"  benchmark to beat                 : {s['benchmark_total']:.6f}")
    print(f"  margin                            : {s['margin_vs_benchmark']:+.6f}")
    print(f"  beats benchmark                   : {s['beats_benchmark']}")
    fv = s["protocol"]["fast_objective_validation"]
    print(f"  fast-objective max deviation      : inner {fv['inner']['max_abs_deviation']:.2e}, "
          f"val {fv['val']['max_abs_deviation']:.2e} (gate {fv['inner']['tol']:.0e})")
    print("\n=== sensitivities ===")
    for k, v in s.get("sensitivity", {}).items():
        print(f"  {k}: {v['total_score']:.6f}")


if __name__ == "__main__":
    main()
