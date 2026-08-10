"""Re-render the Step-3 figures from the persisted artefacts.

The figure functions live next to the analyses that produced them, but they read
only from ``results/``, so the plots can be restyled without re-running any
prediction or scoring pass.  Used after adjusting axis formatting.
"""

from __future__ import annotations

from repo_paths import WORKFLOW_DIR

import importlib.util
import json
import sys

import pandas as pd

WF = str(WORKFLOW_DIR)
sys.path.insert(0, WF)

from common import RESULTS  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, f"{WF}/{path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    E = _load("eval16", "16_eval_gbdt.py")
    S = _load("sens19", "19_round_sensitivity.py")

    sc = json.loads((RESULTS / "gbdt_model_scores.json").read_text(encoding="utf-8"))
    # make_performance_figure needs the module-score/weight tree per predictor.
    results = {nm: {"total_score": sc["model_totals"][nm],
                    "module_scores": sc["model_module_scores"][nm],
                    "module_weights": sc["models"][nm]["module_weights"]}
               for nm in sc["model_totals"]}
    spl = pd.read_csv(RESULTS / "gbdt_split_metrics.csv")
    bench = float(sc["success_gate"]["benchmark_total_score"])
    best = str(sc["success_gate"]["best_model"])

    E.make_performance_figure(results, bench, spl, best)
    E.make_importance_figure()

    df = pd.read_csv(RESULTS / "gbdt_round_sensitivity.csv")
    S.make_figure(df, bench)
    print("regenerated 3 Step-3 figure pairs (png + pdf)", flush=True)


if __name__ == "__main__":
    main()
