"""Diagnostic: where exactly does the 0.4454 benchmark beat our best model?

Prints a per-module table of module score x weight so Step 4 effort can be
aimed at the modules that actually move the total, rather than spread evenly.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
from pathlib import Path

SESSION = REPO_ROOT
RESULTS = SESSION / "results"


def main() -> None:
    scores = json.loads((RESULTS / "gbdt_model_scores.json").read_text(encoding="utf-8"))

    print(f"top-level keys: {list(scores)}\n")

    # Find every dict anywhere in the payload that carries a 'module_scores'
    # block, and key it by the name of its parent mapping entry.
    rows: dict[str, dict] = {}
    weights = None

    def walk(node, name=None):
        nonlocal weights
        if isinstance(node, dict):
            if "module_scores" in node and isinstance(node["module_scores"], dict):
                if name:
                    rows[name] = node["module_scores"]
                weights = node.get("module_weights", weights)
            for k, v in node.items():
                walk(v, k if isinstance(v, dict) else None)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(scores)
    print(f"found {len(rows)} predictors with module_scores")
    if not rows:
        print("no module_scores found; dumping structure")
        print(json.dumps(scores, indent=1)[:3000])
        return

    mods = list(weights)
    focus = [
        "per_context_mean_batch",
        "blend_ens_delta_benchmark_w0.5",
        "ens_delta",
        "xgb_delta",
        "lgb_delta",
        "control_anchor",
    ]
    focus = [f for f in focus if f in rows]

    w = {m: weights[m] for m in mods}
    hdr = f"{'module':18s} {'wt':>5s} " + " ".join(f"{n[:14]:>15s}" for n in focus)
    print(hdr)
    print("-" * len(hdr))
    for m in mods:
        line = f"{m:18s} {w[m]:5.2f} " + " ".join(f"{rows[n].get(m, float('nan')):15.4f}" for n in focus)
        print(line)
    print("-" * len(hdr))
    tot = {n: sum(w[m] * rows[n].get(m, 0.0) for m in mods) for n in focus}
    print(f"{'TOTAL':18s} {1.0:5.2f} " + " ".join(f"{tot[n]:15.4f}" for n in focus))

    print("\n=== weighted contribution (weight x score) ===")
    print(hdr)
    for m in mods:
        line = f"{m:18s} {w[m]:5.2f} " + " ".join(
            f"{w[m] * rows[n].get(m, float('nan')):15.4f}" for n in focus
        )
        print(line)

    bench, best = "per_context_mean_batch", "blend_ens_delta_benchmark_w0.5"
    if bench in rows and best in rows:
        print(f"\n=== per-module deficit of {best} vs {bench} (weighted) ===")
        deficits = []
        for m in mods:
            d = w[m] * (rows[best].get(m, 0.0) - rows[bench].get(m, 0.0))
            deficits.append((d, m))
        for d, m in sorted(deficits):
            flag = "  <-- LOSS" if d < -0.001 else ("  <-- gain" if d > 0.001 else "")
            print(f"  {m:18s} {d:+.5f}{flag}")
        print(f"  {'NET':18s} {sum(d for d, _ in deficits):+.5f}")

        print("\n=== headroom: max achievable by taking the better model per module ===")
        # legitimate only where a module's rows are disjoint from other modules'
        oracle = sum(w[m] * max(rows[best].get(m, 0.0), rows[bench].get(m, 0.0)) for m in mods)
        print(f"  per-module oracle over these two predictors: {oracle:.6f}")
        print(f"  benchmark: {tot[bench]:.6f}   best single: {tot[best]:.6f}")
        print(
            "  NOTE: m1/m2/m4 share all_val rows, so a per-module oracle is NOT directly\n"
            "        realisable by one matrix; the m3_* modules use disjoint splits and ARE\n"
            "        separately addressable via regime-specific blending."
        )


if __name__ == "__main__":
    main()
