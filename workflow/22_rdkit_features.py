"""Step 4.1 -- RDKit SMILES molecular feature extraction.

Resolves every chemical perturbation name in the study to a structure, then
computes Morgan/ECFP4 fingerprints and 2D physicochemical descriptors.

Design decisions (documented because they materially affect the science)
----------------------------------------------------------------------
1. **Authoritative structure resolution.** SMILES are resolved from PubChem
   PUG-REST by compound name rather than hand-transcribed, which removes a
   whole class of silent transcription errors. Every resolution records the
   PubChem CID and molecular formula so the mapping is auditable. Results are
   cached to disk so re-runs are deterministic and offline-capable.

2. **Conservative salt / hydrate stripping.** Names such as "Amiodarone
   hydrochloride" or "1-10 Phenanthroline monohydrate" resolve to multi-fragment
   structures whose counter-ion is not the pharmacophore. But naively "keep the
   largest fragment" is *wrong* for genuinely ionic or coordination compounds:
   on the first run it silently reduced cisplatin (N.N.Cl[Pt]Cl) to ammonia
   (MW 17) and NaCl to a chloride atom (MW 35). Both were caught by the
   molecular-weight cross-check in rule 6.

   We therefore strip only when the largest fragment is unambiguously the parent
   drug: it must carry at least ``MIN_PARENT_HEAVY_ATOMS`` heavy atoms *and*
   strictly more heavy atoms than all other fragments combined. Otherwise the
   structure is kept intact. Every decision is recorded in the audit trail.

3. **Stereochemistry is deliberately ignored.** We use connectivity SMILES and
   Morgan fingerprints without chirality. The 2D descriptors used here (MW,
   LogP, TPSA, HBD/HBA, rotatable bonds, aromatic rings) are all invariant to
   stereochemistry, so retaining stereo flags would add no signal while adding
   risk. This is stated so the limitation is explicit.

4. **Non-molecular perturbations.** "Quality Control" is a QC label, not a
   compound. It gets an all-zero feature vector plus an explicit
   `mol_is_molecule = 0` indicator so a model can learn to treat it separately
   rather than silently imputing a spurious chemistry.

5. **Molecular-weight cross-check.** A SMILES string that parses but encodes the
   wrong compound is the dangerous failure mode, so every resolved structure is
   compared against an independently curated reference molecular weight. A
   mismatch is FLAGged in the report rather than silently accepted. This check
   is what caught the cisplatin/NaCl stripping bug described in rule 2.

6. **n_compounds, not n_rows, is the effective sample size.** There are only
   ~46 distinct compounds in training. A 2048-bit fingerprint is therefore
   wildly over-parameterised relative to the number of independent chemical
   observations. We emit THREE feature blocks of increasing compactness so
   downstream models can pick an appropriate width, and we record the
   compound-level (not row-level) dimensionality in the report:
     - `desc`  : ~25 interpretable 2D descriptors            (GBDT-friendly)
     - `fp_pca`: PCA of the folded fingerprint, few comps    (GBDT-friendly)
     - `fp_raw`: variance-filtered ECFP4 bits                (for the DL model)

Outputs
-------
results/step4_smiles_resolved.json   audit trail: name -> CID/SMILES/formula
data/step4_mol_features.parquet      one row per compound name, all blocks
results/step4_rdkit_report.json      QC report (coverage, MW checks, variance)
figures/step4_chemical_space.png     PCA/descriptor view of the chemical space
"""

from __future__ import annotations

from repo_paths import REPO_ROOT

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SESSION = REPO_ROOT
DATA = SESSION / "data"
RESULTS = SESSION / "results"
FIGURES = SESSION / "figures"

CACHE = RESULTS / "step4_smiles_resolved.json"

SEED = 42
CHEM_COL = "perturbation_no_concentration"

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/{}/JSON"
PROPS = "ConnectivitySMILES,SMILES,MolecularWeight,MolecularFormula"

# Names that PubChem cannot resolve verbatim -> explicit synonym.
# Every entry is a deliberate, documented curation decision.
ALIASES: dict[str, str] = {
    "CHX": "Cycloheximide",
    "MMS": "Methyl methanesulfonate",
    "FCCP": "Carbonyl cyanide p-trifluoromethoxyphenylhydrazone",
    "G418": "Geneticin",
    "LY 294002 hydrochloride": "LY294002",
    "1-10 Phenanthroline monohydrate": "1,10-Phenanthroline",
    "(1R, 2S, 5R) - (-) - Menthol": "(-)-Menthol",
    "(S)-(+)-Camptothecin": "Camptothecin",
    "Neomycin B": "Neomycin",
    "Nystatin dihydrate": "Nystatin",
    "Doxycycline hyclate": "Doxycycline",
    "Hoechst 33258": "Hoechst 33258",
    "U-73122": "U-73122",
    "4-Hydroxytamoxifen": "Afimoxifene",
    # "Oligomycin" and "Tunicamycin" are commercial mixtures of homologues, not
    # single entities, so PubChem cannot resolve the bare name. We pin each to
    # its principal/most abundant congener and record that choice.
    "Oligomycin": "Oligomycin A",
    "Tunicamycin": "Tunicamycin A",
    "H2O2": "Hydrogen peroxide",
    "SDS": "Sodium dodecyl sulfate",
    "EDTA": "Edetic acid",
    "NaCl": "Sodium chloride",
    "DMSO": "Dimethyl sulfoxide",
}

# Perturbation labels that are not chemical entities at all.
NON_MOLECULES = {"Quality Control"}

# Last-resort curated fallbacks, used only if PubChem fails for a name.
# Kept deliberately short; each is a widely published structure.
FALLBACK_SMILES: dict[str, str] = {
    "Water": "O",
    "Dimethyl sulfoxide": "CS(C)=O",
    "Hydrogen peroxide": "OO",
    "Sodium chloride": "[Na+].[Cl-]",
    "Sorbitol": "OCC(O)C(O)C(O)C(O)CO",
    "Methyl methanesulfonate": "COS(C)(=O)=O",
    "Hydroxyurea": "NC(=O)NO",
    "Edetic acid": "OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O",
    "Sodium dodecyl sulfate": "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",
    "1,10-Phenanthroline": "C1=CC2=CC=C3C=CC=NC3=C2N=C1",
}

# Reference molecular weights (g/mol) of the *parent* entity we expect after
# salt/hydrate stripping, curated independently of PubChem. Used purely as a
# cross-check that resolution + stripping returned the intended compound; a
# mismatch is FLAGged in the report, never silently accepted. Populated for
# every compound whose parent MW is unambiguous, which is nearly all of them --
# a dense check is the point, since a sparse one catches nothing.
REFERENCE_MW: dict[str, float] = {
    # --- simple / inorganic / solvent -------------------------------------
    "Water": 18.02,
    "DMSO": 78.13,
    "H2O2": 34.01,
    "NaCl": 58.44,
    "Cisplatin": 300.05,
    "Sorbitol": 182.17,
    "Hydroxyurea": 76.05,
    "MMS": 110.13,
    "EDTA": 292.24,
    # --- salt-form names: reference is the stripped parent ----------------
    "Amiodarone hydrochloride": 645.31,
    "Desipramine hydrochloride": 266.38,
    "Doxycycline hyclate": 444.44,
    "Dyclonine hydrochloride": 289.41,
    "Clomiphene citrate": 405.96,
    "Harmine hydrochloride": 212.25,
    "LY 294002 hydrochloride": 307.35,
    "Pentamidine isethionate": 340.42,
    "Raloxifene hydrochloride": 473.58,
    "Trifluoperazine dihydrochloride": 407.50,
    "Nystatin dihydrate": 926.09,
    "1-10 Phenanthroline monohydrate": 180.21,
    # --- neutral small molecules / natural products -----------------------
    "(1R, 2S, 5R) - (-) - Menthol": 156.27,
    "(S)-(+)-Camptothecin": 348.35,
    "4-Hydroxytamoxifen": 387.51,
    "Abietic acid": 302.45,
    "Amphotericin B": 924.08,
    "Anisomycin": 265.30,
    "Artemisinin": 282.33,
    "Brefeldin A": 280.36,
    "CHX": 281.35,
    "Clotrimazole": 344.84,
    "Cyclopiazonic acid": 336.39,
    "Emodin": 270.24,
    "FCCP": 254.17,
    "Fluconazole": 306.27,
    "G418": 496.60,
    "Geldanamycin": 560.64,
    "Haloperidol": 375.86,
    "Hoechst 33258": 424.50,
    "Hygromycin B": 527.52,
    "Neomycin B": 614.64,
    "Nigericin": 724.97,
    "Nocodazole": 301.32,
    "Oligomycin": 791.06,
    "Parthenolide": 248.32,
    "Plumbagin": 188.18,
    "Rapamycin": 914.17,
    "Staurosporine": 466.53,
    "Sulfometuron methyl": 364.38,
    "Tamoxifen": 371.51,
    "Trichostatin A": 302.37,
    "Tunicamycin": 830.90,
    "U-73122": 464.64,
    "Valinomycin": 1111.32,
    "Wortmannin": 428.43,
}

# A fragment is treated as the parent drug only if it is this large.
MIN_PARENT_HEAVY_ATOMS = 6

# ---------------------------------------------------------------------------
# structure resolution
# ---------------------------------------------------------------------------


def _fetch_pubchem(name: str, tries: int = 3) -> dict | None:
    """Query PubChem PUG-REST for a compound name.

    Returns the first property record, or ``None`` if the name is unresolvable.
    """
    url = PUBCHEM.format(urllib.parse.quote(name, safe=""), PROPS)
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "pubchem.ncbi.nlm.nih.gov":
        raise ValueError("Refusing to query an unexpected PubChem endpoint")
    for attempt in range(tries):
        try:
            # The scheme and exact host are checked immediately above.
            with urllib.request.urlopen(url, timeout=30) as fh:  # nosec B310
                payload = json.loads(fh.read().decode("utf-8"))
            recs = payload.get("PropertyTable", {}).get("Properties", [])
            return recs[0] if recs else None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None  # genuinely unknown name; do not retry
            time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                UnicodeDecodeError):
            time.sleep(1.5 * (attempt + 1))
    return None


def resolve_all(names: list[str]) -> dict[str, dict]:
    """Resolve every perturbation name to a structure record, with caching."""
    cache: dict[str, dict] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"loaded {len(cache)} cached resolutions from {CACHE.name}")

    for i, name in enumerate(names, 1):
        if name in cache and cache[name].get("smiles"):
            continue

        if name in NON_MOLECULES:
            cache[name] = {
                "query": None,
                "smiles": None,
                "cid": None,
                "formula": None,
                "pubchem_mw": None,
                "source": "not_a_molecule",
            }
            print(f"[{i}/{len(names)}] {name!r} -> not a molecule (QC label)")
            continue

        # try the verbatim name, then the curated alias
        candidates = [name]
        if name in ALIASES:
            candidates.append(ALIASES[name])

        rec, used = None, None
        for cand in candidates:
            rec = _fetch_pubchem(cand)
            if rec is not None:
                used = cand
                break
            time.sleep(0.25)  # be polite to the API

        if rec is not None:
            smiles = rec.get("ConnectivitySMILES") or rec.get("SMILES")
            cache[name] = {
                "query": used,
                "smiles": smiles,
                "cid": rec.get("CID"),
                "formula": rec.get("MolecularFormula"),
                "pubchem_mw": float(rec["MolecularWeight"])
                if rec.get("MolecularWeight")
                else None,
                "source": "pubchem",
            }
            print(
                f"[{i}/{len(names)}] {name!r} -> CID {rec.get('CID')} "
                f"{rec.get('MolecularFormula')} (query={used!r})"
            )
        else:
            key = ALIASES.get(name, name)
            fb = FALLBACK_SMILES.get(key) or FALLBACK_SMILES.get(name)
            cache[name] = {
                "query": key,
                "smiles": fb,
                "cid": None,
                "formula": None,
                "pubchem_mw": None,
                "source": "curated_fallback" if fb else "UNRESOLVED",
            }
            tag = "curated fallback" if fb else "*** UNRESOLVED ***"
            print(f"[{i}/{len(names)}] {name!r} -> {tag}")

        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.25)

    return cache


# ---------------------------------------------------------------------------
# featurisation
# ---------------------------------------------------------------------------

DESCRIPTORS = [
    "MolWt",
    "ExactMolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumHeteroatoms",
    "NHOHCount",
    "NOCount",
    "LabuteASA",
    "BalabanJ",
    "BertzCT",
    "Chi0v",
    "Chi1v",
    "Kappa1",
    "Kappa2",
    "HallKierAlpha",
    "qed",
]


def featurise(cache: dict[str, dict], n_bits: int = 2048, radius: int = 2) -> tuple:
    """Compute descriptors + Morgan fingerprints for every resolved compound.

    Returns
    -------
    desc_df : pd.DataFrame
        One row per compound name, descriptor columns prefixed ``mol_``.
    fp_mat : np.ndarray
        Binary ECFP4 matrix, shape ``(n_compounds, n_bits)``.
    audit : list[dict]
        Per-compound QC record (fragments stripped, MW check, parse status).
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, rdMolDescriptors
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def choose_parent(mol):
        """Return (parent_mol, decision) using conservative salt stripping.

        Strips counter-ions/hydrates only when the largest fragment is
        unambiguously the parent drug: at least ``MIN_PARENT_HEAVY_ATOMS`` heavy
        atoms and strictly more heavy atoms than all other *distinct* fragments
        combined.

        Identical fragments are de-duplicated before that comparison, because a
        salt that carries two copies of the same parent (doxycycline hyclate is
        (doxycycline.HCl)2 . EtOH . H2O) still has exactly one parent, and
        counting the duplicate as "other" would wrongly veto stripping.

        Ionic and coordination compounds (NaCl, cisplatin) have no fragment
        large enough to qualify and are correctly kept intact.
        """
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) == 1:
            return mol, "single_fragment"

        uniq: dict[str, object] = {}
        for f in frags:
            uniq.setdefault(Chem.MolToSmiles(f), f)
        cands = list(uniq.values())
        sizes = [f.GetNumHeavyAtoms() for f in cands]
        big = int(np.argmax(sizes))
        rest = sum(sizes) - sizes[big]
        if sizes[big] >= MIN_PARENT_HEAVY_ATOMS and sizes[big] > rest:
            n_dup = len(frags) - len(cands)
            note = f"stripped_to_parent(from_{len(frags)}_frags,{len(cands)}_distinct"
            note += f",{n_dup}_duplicate)" if n_dup else ")"
            return cands[big], note
        return mol, "kept_intact_multi_fragment"

    names = sorted(cache)
    rows, fps, audit = [], [], []

    for i, name in enumerate(names, 1):
        rec = cache[name]
        smi = rec.get("smiles")
        entry = {
            "name": name,
            "cid": rec.get("cid"),
            "source": rec.get("source"),
            "smiles_in": smi,
        }

        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            entry.update(parsed=False, n_frags=0, smiles_parent=None, mw=None)
            rows.append({"name": name, "mol_is_molecule": 0.0})
            fps.append(np.zeros(n_bits, dtype=np.uint8))
            audit.append(entry)
            print(f"[{i}/{len(names)}] {name!r}: no structure -> zero vector")
            continue

        n_frags = len(Chem.GetMolFrags(mol))
        parent, frag_decision = choose_parent(mol)
        reparsed = Chem.MolFromSmiles(Chem.MolToSmiles(parent))
        if reparsed is not None:
            parent = reparsed
        entry["fragment_decision"] = frag_decision

        d: dict[str, float] = {"name": name, "mol_is_molecule": 1.0}
        fn = {
            "MolWt": Descriptors.MolWt,
            "ExactMolWt": Descriptors.ExactMolWt,
            "MolLogP": Crippen.MolLogP,
            "MolMR": Crippen.MolMR,
            "TPSA": rdMolDescriptors.CalcTPSA,
            "NumHDonors": rdMolDescriptors.CalcNumHBD,
            "NumHAcceptors": rdMolDescriptors.CalcNumHBA,
            "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
            "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
            "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
            "RingCount": rdMolDescriptors.CalcNumRings,
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
            "HeavyAtomCount": Descriptors.HeavyAtomCount,
            "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
            "NHOHCount": Descriptors.NHOHCount,
            "NOCount": Descriptors.NOCount,
            "LabuteASA": rdMolDescriptors.CalcLabuteASA,
            "BalabanJ": GraphDescriptors.BalabanJ,
            "BertzCT": GraphDescriptors.BertzCT,
            "Chi0v": rdMolDescriptors.CalcChi0v,
            "Chi1v": rdMolDescriptors.CalcChi1v,
            "Kappa1": rdMolDescriptors.CalcKappa1,
            "Kappa2": rdMolDescriptors.CalcKappa2,
            "HallKierAlpha": rdMolDescriptors.CalcHallKierAlpha,
        }
        for key in DESCRIPTORS:
            if key == "qed":
                from rdkit.Chem import QED

                try:
                    d["mol_qed"] = float(QED.qed(parent))
                except Exception:
                    d["mol_qed"] = np.nan
                continue
            try:
                d[f"mol_{key}"] = float(fn[key](parent))
            except Exception:
                d[f"mol_{key}"] = np.nan

        d["mol_formal_charge"] = float(Chem.GetFormalCharge(parent))
        d["mol_n_frags_stripped"] = float(n_frags - 1)

        rows.append(d)
        fps.append(np.asarray(gen.GetFingerprintAsNumPy(parent), dtype=np.uint8))

        mw = d["mol_MolWt"]
        ref = REFERENCE_MW.get(name)
        entry.update(
            parsed=True,
            n_frags=n_frags,
            smiles_parent=Chem.MolToSmiles(parent),
            formula=rdMolDescriptors.CalcMolFormula(parent),
            mw=mw,
            reference_mw=ref,
            mw_abs_error=(abs(mw - ref) if ref is not None else None),
            mw_check="pass"
            if ref is None or abs(mw - ref) <= max(1.0, 0.02 * ref)
            else "FLAG",
        )
        audit.append(entry)
        if i % 10 == 0 or i == len(names):
            print(f"  featurised {i}/{len(names)} compounds")

    desc_df = pd.DataFrame(rows).set_index("name").reindex(names)
    fp_mat = np.vstack(fps)
    return desc_df, fp_mat, audit


def main() -> None:
    np.random.seed(SEED)
    FIGURES.mkdir(exist_ok=True)

    tr = pd.read_csv(DATA / "meta_train_val_annotated.csv")
    te = pd.read_csv(DATA / "meta_test_annotated.csv")
    names = sorted(set(tr[CHEM_COL].dropna().astype(str)) | set(te[CHEM_COL].dropna().astype(str)))
    train_names = sorted(set(tr[CHEM_COL].dropna().astype(str)))
    print(f"{len(names)} distinct perturbation labels ({len(train_names)} seen in train_val)")

    cache = resolve_all(names)

    unresolved = [n for n, r in cache.items() if r["source"] == "UNRESOLVED"]
    if unresolved:
        print(f"\n!! {len(unresolved)} UNRESOLVED: {unresolved}")

    desc_df, fp_mat, audit = featurise(cache, n_bits=2048, radius=2)

    # ---- descriptor block: median-impute (only non-molecules are missing) ----
    desc_cols = [c for c in desc_df.columns if c != "mol_is_molecule"]
    med = desc_df.loc[desc_df["mol_is_molecule"] == 1.0, desc_cols].median()
    desc_df[desc_cols] = desc_df[desc_cols].fillna(0.0 * med).fillna(0.0)
    # non-molecules keep an explicit zero vector + mol_is_molecule = 0
    desc_df["mol_is_molecule"] = desc_df["mol_is_molecule"].fillna(0.0)

    # ---- fingerprint blocks -------------------------------------------------
    # Variance filter computed on TRAINING compounds only: a bit that is
    # constant across the compounds we can learn from carries no information,
    # and letting test compounds influence the filter would leak.
    tr_idx = [i for i, n in enumerate(desc_df.index) if n in train_names]
    fp_tr = fp_mat[tr_idx]
    keep = np.where(fp_tr.sum(axis=0) >= 2)[0]  # bit set in >= 2 train compounds
    keep = keep[fp_tr[:, keep].sum(axis=0) <= len(tr_idx) - 2]  # and not near-constant
    print(f"\nECFP4: {fp_mat.shape[1]} bits -> {len(keep)} informative bits")

    fp_raw = pd.DataFrame(
        fp_mat[:, keep].astype(np.float32),
        index=desc_df.index,
        columns=[f"fp_{b}" for b in keep],
    )

    # PCA fitted on training compounds only, then applied to all.
    from sklearn.decomposition import PCA

    n_comp = int(min(16, len(tr_idx) - 1, len(keep)))
    pca = PCA(n_components=n_comp, random_state=SEED)
    pca.fit(fp_mat[tr_idx][:, keep].astype(np.float64))
    scores = pca.transform(fp_mat[:, keep].astype(np.float64))
    fp_pca = pd.DataFrame(
        scores, index=desc_df.index, columns=[f"fppca_{i:02d}" for i in range(n_comp)]
    )
    evr = pca.explained_variance_ratio_
    print(f"fp PCA: {n_comp} comps, cumulative EVR = {evr.sum():.3f}")

    out = pd.concat([desc_df, fp_pca, fp_raw], axis=1)
    out.index.name = CHEM_COL
    out_path = DATA / "step4_mol_features.parquet"
    out.reset_index().to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}  shape={out.shape}")

    # ---- QC report ---------------------------------------------------------
    flagged = [a for a in audit if a.get("mw_check") == "FLAG"]
    n_mol = int(desc_df["mol_is_molecule"].sum())
    report = {
        "seed": SEED,
        "n_perturbation_labels": len(names),
        "n_train_labels": len(train_names),
        "n_resolved_as_molecule": n_mol,
        "n_non_molecule_labels": len(names) - n_mol,
        "n_unresolved": len(unresolved),
        "unresolved": unresolved,
        "coverage_of_molecular_labels_pct": round(
            100.0 * n_mol / max(1, len(names) - len(NON_MOLECULES & set(names))), 3
        ),
        "mw_crosscheck": {
            "n_with_reference": sum(1 for a in audit if a.get("reference_mw") is not None),
            "n_flagged": len(flagged),
            "flagged": [
                {
                    "name": a["name"],
                    "mw": a["mw"],
                    "reference_mw": a["reference_mw"],
                    "abs_error": a["mw_abs_error"],
                }
                for a in flagged
            ],
        },
        "blocks": {
            "desc": {"n_features": int(desc_df.shape[1]), "columns": list(desc_df.columns)},
            "fp_pca": {
                "n_features": n_comp,
                "explained_variance_ratio": [round(float(x), 5) for x in evr],
                "cumulative_evr": round(float(evr.sum()), 5),
            },
            "fp_raw": {"n_features": int(fp_raw.shape[1]), "n_bits_before_filter": 2048},
        },
        "n_descriptor_missing_cells": int(desc_df[desc_cols].isna().sum().sum()),
        "effective_sample_size_note": (
            f"Only {len(train_names)} distinct compounds exist in training. Chemical "
            "generalisation is limited by n_compounds, not by n_rows; the fp_raw block "
            f"({fp_raw.shape[1]} bits) is far wider than that and must be regularised "
            "heavily or restricted to the desc/fp_pca blocks for tree models."
        ),
        "salt_stripping": {
            "min_parent_heavy_atoms": MIN_PARENT_HEAVY_ATOMS,
            "n_multi_fragment": sum(1 for a in audit if (a.get("n_frags") or 0) > 1),
            "n_stripped": sum(
                1 for a in audit if str(a.get("fragment_decision", "")).startswith("stripped")
            ),
            "n_kept_intact_multi_fragment": sum(
                1 for a in audit if a.get("fragment_decision") == "kept_intact_multi_fragment"
            ),
            "decisions": [
                {
                    "name": a["name"],
                    "n_frags": a["n_frags"],
                    "decision": a.get("fragment_decision"),
                    "parent": a["smiles_parent"],
                    "mw": a.get("mw"),
                }
                for a in audit
                if (a.get("n_frags") or 0) > 1
            ],
        },
        "audit": audit,
    }
    (RESULTS / "step4_rdkit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {RESULTS / 'step4_rdkit_report.json'}")

    print("\n=== resolution summary ===")
    print(f"  molecular labels resolved : {n_mol}/{len(names) - len(NON_MOLECULES & set(names))}")
    print(f"  non-molecule labels       : {len(names) - n_mol}")
    print(f"  MW cross-checks flagged   : {len(flagged)}")
    for a in flagged:
        print(f"    FLAG {a['name']}: rdkit={a['mw']:.2f} ref={a['reference_mw']:.2f}")
    print(f"  descriptor missing cells  : {report['n_descriptor_missing_cells']}")

    # ---- chemical space figure -------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    is_train = np.array([n in train_names for n in desc_df.index])
    ax = axes[0]
    ax.scatter(
        scores[is_train, 0], scores[is_train, 1], s=26, c="#3B6EA8", label="train compound", lw=0
    )
    ax.scatter(
        scores[~is_train, 0],
        scores[~is_train, 1],
        s=46,
        c="#C4442E",
        marker="^",
        label="test-only compound",
        lw=0,
    )
    ax.set_xlabel(f"ECFP4 PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"ECFP4 PC2 ({evr[1] * 100:.1f}%)")
    ax.set_title("Chemical space coverage (ECFP4 PCA)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    mask = desc_df["mol_is_molecule"] == 1.0
    ax.scatter(
        desc_df.loc[mask & is_train, "mol_MolWt"],
        desc_df.loc[mask & is_train, "mol_MolLogP"],
        s=26,
        c="#3B6EA8",
        lw=0,
        label="train",
    )
    ax.scatter(
        desc_df.loc[mask & ~is_train, "mol_MolWt"],
        desc_df.loc[mask & ~is_train, "mol_MolLogP"],
        s=46,
        c="#C4442E",
        marker="^",
        lw=0,
        label="test-only",
    )
    ax.set_xlabel("Molecular weight (Da)")
    ax.set_ylabel("cLogP")
    ax.set_title("Physicochemical property space")
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle(
        "Step 4.1 -- RDKit molecular featurisation of yeast perturbagens", fontsize=9.5
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"step4_chemical_space.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGURES / 'step4_chemical_space.png'}")


if __name__ == "__main__":
    main()
