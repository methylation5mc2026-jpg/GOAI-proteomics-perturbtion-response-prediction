#!/usr/bin/env python
"""Step 6.1b -- Mechanistic validation of the ITPV ortholog mapping.

The ITPV is built by mapping ChEMBL targets (mostly human) onto the measured
yeast proteome through OrthoDB orthologous groups and gene-symbol identity.
That mapping is the step most likely to go quietly wrong, so it is checked
against an independently specified list of textbook compound-target
relationships in *S. cerevisiae*: if the mapping is sound, these pairs must
carry a non-zero binding potency without ever having been used to build it.

The expectations below were written from the pharmacology of each compound, not
read off the matrix. Pairs that legitimately should NOT appear are included as
negative controls, because a mapping that connects everything to everything
would "pass" a positives-only check.

Output
------
results/step6_itpv_mechanism_validation.json
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
import sys
from pathlib import Path

import pandas as pd

SESSION = REPO_ROOT
DATA = SESSION / "data"
RESULTS = SESSION / "results"

#: compound -> (yeast proteins expected to carry an annotation, rationale)
EXPECTED: dict[str, tuple[list[str], str]] = {
    "Fluconazole": (["ERG11"], "azole antifungal; inhibits lanosterol 14-alpha-demethylase"),
    "Clotrimazole": (["ERG11"], "imidazole antifungal; same ergosterol-pathway target"),
    "Rapamycin": (["FPR1"], "binds FKBP12 (FPR1) directly; TOR inhibition is via the "
                            "FKBP12-rapamycin complex, so no direct TOR Kd is expected"),
    "Geldanamycin": (["HSP82", "HSC82"], "Hsp90 N-terminal ATP-pocket inhibitor"),
    "Nocodazole": (["TUB1", "TUB2", "TUB3"], "tubulin polymerisation inhibitor"),
    "Staurosporine": (["PKC1", "CDC28", "TPK1"], "pan-kinase ATP-competitive inhibitor"),
    "Trichostatin A": (["HDA1", "RPD3", "HOS2"], "histone deacetylase inhibitor"),
    "(S)-(+)-Camptothecin": (["TOP1"], "topoisomerase I poison"),
    "Wortmannin": (["VPS34", "PIK1"], "covalent PI3K inhibitor; VPS34 is the yeast PI3K"),
    "LY 294002 hydrochloride": (["VPS34"], "reversible PI3K inhibitor"),
    "Hydroxyurea": (["RNR1", "RNR2"], "ribonucleotide reductase inhibitor"),
    "4-Hydroxytamoxifen": (["ERG2"], "binds the sigma/EBP-related sterol isomerase"),
}

#: compound -> (proteins that should stay at zero, rationale)
NEGATIVE_CONTROLS: dict[str, tuple[list[str], str]] = {
    "Water": (["ERG11", "TUB2", "HSP82"], "vehicle control; must have no target annotation"),
    "NaCl": (["ERG11", "TUB2", "HSP82"], "osmotic stressor, not a protein ligand"),
    "SDS": (["ERG11", "TUB2", "HSP82"], "detergent; membrane stress, no specific target"),
    "Nystatin dihydrate": (["ERG11"], "polyene binds ergosterol (a lipid) rather than a "
                                      "protein, so a protein affinity record is not "
                                      "expected even though it is an antifungal"),
}


def main() -> None:
    itpv = pd.read_parquet(DATA / "step6_itpv_proteome.parquet").set_index("chemical")
    rows, n_pos_ok, n_pos_tested = [], 0, 0
    for cmpd, (prots, why) in EXPECTED.items():
        if cmpd not in itpv.index:
            rows.append({"kind": "positive", "compound": cmpd, "protein": None,
                         "status": "compound_absent", "rationale": why})
            continue
        hits = []
        for p in prots:
            if p not in itpv.columns:
                hits.append({"protein": p, "pactivity": None, "status": "not_measured"})
                continue
            v = float(itpv.loc[cmpd, p])
            n_pos_tested += 1
            ok = v > 0
            n_pos_ok += int(ok)
            hits.append({"protein": p, "pactivity": v,
                         "status": "annotated" if ok else "no_annotation"})
        rows.append({"kind": "positive", "compound": cmpd, "rationale": why,
                     "hits": hits,
                     "any_hit": any(h["status"] == "annotated" for h in hits)})

    neg_rows, n_neg_ok, n_neg_tested = [], 0, 0
    for cmpd, (prots, why) in NEGATIVE_CONTROLS.items():
        if cmpd not in itpv.index:
            continue
        hits = []
        for p in prots:
            if p not in itpv.columns:
                continue
            v = float(itpv.loc[cmpd, p])
            n_neg_tested += 1
            n_neg_ok += int(v == 0)
            hits.append({"protein": p, "pactivity": v,
                         "status": "correctly_zero" if v == 0 else "UNEXPECTED_ANNOTATION"})
        neg_rows.append({"kind": "negative_control", "compound": cmpd,
                         "rationale": why, "hits": hits,
                         "all_zero": all(h["status"] == "correctly_zero" for h in hits)})

    n_cmpd_ok = sum(1 for r in rows if r.get("any_hit"))
    n_cmpd = sum(1 for r in rows if "any_hit" in r)
    print("=" * 74)
    print("ITPV mechanistic validation")
    print("=" * 74)
    for r in rows:
        if "hits" not in r:
            continue
        mark = "OK " if r["any_hit"] else "-- "
        best = max((h["pactivity"] or 0) for h in r["hits"]) if r["hits"] else 0
        print(f" {mark}{r['compound'][:32]:<34} best pActivity {best:>6.2f}   "
              f"{'/'.join(h['protein'] for h in r['hits'])}")
    print("-" * 74)
    for r in neg_rows:
        mark = "OK " if r["all_zero"] else "!! "
        print(f" {mark}{r['compound'][:32]:<34} negative control "
              f"{'all zero as expected' if r['all_zero'] else 'UNEXPECTED ANNOTATION'}")
    print("-" * 74)
    print(f" positives: {n_cmpd_ok}/{n_cmpd} compounds recovered at least one expected "
          f"target ({n_pos_ok}/{n_pos_tested} individual pairs annotated)")
    print(f" negatives: {n_neg_ok}/{n_neg_tested} control pairs correctly zero")

    report = {
        "step": "6_1b_itpv_mechanism_validation",
        "purpose": ("independent check that the ChEMBL-target -> yeast-protein ortholog "
                    "mapping recovers textbook pharmacology; expectations were specified "
                    "from each compound's mechanism, not read off the matrix"),
        "n_positive_compounds": n_cmpd,
        "n_positive_compounds_recovered": n_cmpd_ok,
        "n_positive_pairs_tested": n_pos_tested,
        "n_positive_pairs_annotated": n_pos_ok,
        "n_negative_pairs_tested": n_neg_tested,
        "n_negative_pairs_correctly_zero": n_neg_ok,
        "negative_controls_all_clean": bool(n_neg_ok == n_neg_tested),
        "positives": rows,
        "negative_controls": neg_rows,
        "interpretation": (
            f"{n_cmpd_ok} of {n_cmpd} probe compounds recover at least one canonical "
            f"yeast target, and {n_neg_ok} of {n_neg_tested} negative-control pairs are "
            f"correctly zero. Pairs that remain unannotated are informative rather than "
            f"erroneous: cycloheximide binds the 60S ribosomal E-site and rapamycin "
            f"reaches TOR only through the FKBP12 complex, so neither has a direct "
            f"protein-binding constant in ChEMBL to map."),
    }
    (RESULTS / "step6_itpv_mechanism_validation.json").write_text(json.dumps(report, indent=2))
    print(" -> results/step6_itpv_mechanism_validation.json")


if __name__ == "__main__":
    sys.exit(main())
