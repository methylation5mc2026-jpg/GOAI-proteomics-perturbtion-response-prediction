"""Probe: recover the Step-3 per-regime selected config and round count.

The RDKit-enhanced models must be trained at the *same* hyper-parameters as the
Step-3 tabular models, otherwise any change in the S1 score could be a tuning
artefact rather than the effect of the molecular features.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
from pathlib import Path

SESSION = REPO_ROOT
RESULTS = SESSION / "results"


def main() -> None:
    rep = json.loads((RESULTS / "gbdt_training_report.json").read_text(encoding="utf-8"))
    print("top-level keys:", list(rep))
    for k in ("selected", "regime_selection", "selection"):
        if k in rep:
            print(f"\n--- {k} ---")
            print(json.dumps(rep[k], indent=2)[:2500])
    # fall back to a shallow dump of anything mentioning rounds
    if not any(k in rep for k in ("selected", "regime_selection", "selection")):
        print(json.dumps(rep, indent=2)[:4000])

    scores = json.loads((RESULTS / "gbdt_model_scores.json").read_text(encoding="utf-8"))
    if "regime_specialists" in scores:
        print("\n--- regime_specialists (from model scores) ---")
        print(json.dumps(scores["regime_specialists"], indent=2)[:2500])


if __name__ == "__main__":
    main()
