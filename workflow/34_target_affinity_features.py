#!/usr/bin/env python
"""Step 6.1 -- ChEMBL target affinity (MoA) vector mapping.

Builds the Initial Target Perturbation Vector (ITPV): for every chemical
perturbation, a vector over the **5,243 measured yeast proteins** giving the
annotated direct binding potency of that compound against each protein.

Pipeline
--------
1.  Resolve every perturbation to a ChEMBL molecule (InChIKey -> connectivity
    InChIKey -> preferred name -> synonym).
2.  Pull all qualifying activity records (Kd / Ki / IC50 / EC50), convert to
    pActivity = -log10[M], drop censored '>' records.
3.  Resolve each ChEMBL target to its UniProt component accessions.
4.  Map those accessions onto the measured yeast proteome through three
    independently-recorded channels:
      A. *direct*   - the target is already an S. cerevisiae protein.
      B. *orthodb*  - target and yeast protein share a OrthoDB Eukaryota
                      orthologous group (the ortholog channel proper).
      C. *symbol*   - target and yeast protein share a gene symbol / alias.
5.  ITPV[compound, protein] = max pActivity over all targets mapping to that
    protein. Absent pairs are structural zeros.

Scientific notes
----------------
* The ITPV is **sparse by construction and that is the correct behaviour**: only
  a minority of the 5,243 measured proteins are annotated drug targets. Coverage
  is measured and reported rather than inflated. Zero means "no annotated
  engagement", which is a substantive statement; it is not an imputed value.
* Compounds with no ChEMBL evidence at all (solvents, inorganic stressors,
  vehicle controls) carry an all-zero ITPV plus a binary
  ``has_chembl_evidence`` flag, so downstream models can distinguish
  "no annotated target" from "not looked up".
* ChEMBL affinities are external prior knowledge from unrelated assay systems
  and are independent of the yeast proteome response being predicted, so they
  cannot leak the label. The SVD compression is nevertheless fitted on the 37
  LCGO training compounds only, to stay consistent with the Step-5 protocol.

Outputs
-------
data/step6_target_features.parquet          compact model-ready features
data/step6_itpv_proteome.parquet            full compounds x 5243 ITPV
data/step6_chembl_activities.parquet        curated activity records
data/step6_target_protein_map.parquet       target -> yeast protein provenance
results/step6_target_affinity_report.json
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SESSION = REPO_ROOT
DATA = SESSION / "data"
RESULTS = SESSION / "results"
CACHE = DATA / "step6_cache"
CACHE.mkdir(parents=True, exist_ok=True)

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT = "https://rest.uniprot.org/uniprotkb"
YEAST_TAXON = 559292  # S. cerevisiae S288c reference proteome
SEED = 42
np.random.seed(SEED)

QUAL_TYPES = ("Kd", "Ki", "IC50", "EC50")
NON_MOA = {"Quality Control", "Water"}
HDRS = {"Accept": "application/json", "User-Agent": "goai-virtual-cell-step6/1.0"}
ALLOWED_API_HOSTS = {"www.ebi.ac.uk", "rest.uniprot.org"}

SALT_RE = re.compile(
    r"\s*(hydrochloride|dihydrochloride|hyclate|citrate|isethionate|monohydrate|"
    r"dihydrate|sodium|sulfate|maleate|acetate|mesylate|tartrate)\s*$", re.I)


# ---------------------------------------------------------------------------
# HTTP with on-disk cache
# ---------------------------------------------------------------------------
def _cache_path(key: str, ext: str = "json") -> Path:
    return CACHE / (re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:180] + "." + ext)


def _validate_api_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_API_HOSTS:
        raise ValueError(f"Refusing to query an unexpected API endpoint: {url}")


def http_json(url: str, params: dict | None = None, key: str | None = None,
              retries: int = 4, timeout: int = 60):
    _validate_api_url(url)
    if key is not None:
        cp = _cache_path(key)
        if cp.exists():
            try:
                return json.loads(cp.read_text())
            except json.JSONDecodeError:
                cp.unlink(missing_ok=True)
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HDRS, timeout=timeout)
            if r.status_code == 200:
                js = r.json()
                if key is not None:
                    _cache_path(key).write_text(json.dumps(js))
                return js
            if r.status_code == 404:
                if key is not None:
                    _cache_path(key).write_text(json.dumps({"_status": 404}))
                return {"_status": 404}
            last = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5 * (attempt + 1))
    print(f"    [warn] give up {url} {params}: {last}", flush=True)
    return None


def http_text(url: str, params: dict | None = None, key: str | None = None,
              retries: int = 4, timeout: int = 180) -> str | None:
    _validate_api_url(url)
    if key is not None:
        cp = _cache_path(key, "tsv")
        if cp.exists():
            return cp.read_text()
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": HDRS["User-Agent"]})
            if r.status_code == 200:
                if key is not None:
                    _cache_path(key, "tsv").write_text(r.text)
                return r.text
            last = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5 * (attempt + 1))
    print(f"    [warn] give up {url}: {last}", flush=True)
    return None


# ---------------------------------------------------------------------------
# Compound resolution
# ---------------------------------------------------------------------------
def _rank_candidates(mols: list[dict]) -> list[tuple[str, str | None]]:
    """Rank ChEMBL molecule records, preferring canonical parent entries.

    ChEMBL frequently holds several molecule records for one structure: the
    curated parent (low accession number, populated ``pref_name``) plus
    salt/mixture/deposited duplicates that carry no bioactivity. Naively taking
    the API's first hit picks a duplicate often enough to matter -- Rapamycin
    resolved to CHEMBL3560003 and Tamoxifen to CHEMBL2139100, both of which
    return zero activities, instead of the canonical CHEMBL413 / CHEMBL83.
    We therefore map every hit onto its ``molecule_hierarchy.parent_chembl_id``
    and rank parents first, then records with a preferred name, then by
    ascending accession number.
    """
    cands: list[tuple[str, str | None, int, int]] = []
    for m in mols:
        cid = m.get("molecule_chembl_id")
        if not cid:
            continue
        pref = m.get("pref_name")
        hier = m.get("molecule_hierarchy") or {}
        parent = hier.get("parent_chembl_id")
        for c, is_parent in ((parent, 1), (cid, 1 if parent in (None, cid) else 0)):
            if not c:
                continue
            try:
                num = int(str(c).replace("CHEMBL", ""))
            except ValueError:
                num = 10 ** 9
            cands.append((c, pref, is_parent, num))
    seen, out = set(), []
    for c, pref, is_parent, num in sorted(cands, key=lambda t: (-t[2], t[1] is None, t[3])):
        if c in seen:
            continue
        seen.add(c)
        out.append((c, pref))
    return out


def resolve_molecule(name: str, smiles: str | None) -> dict:
    """Resolve a perturbation name/SMILES to ranked ChEMBL molecule ids.

    Returns all plausible candidate ids (canonical parents first). Activities
    are later pooled across candidates, because one structure may be split over
    several ChEMBL records.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    out = {"name": name, "smiles": smiles, "chembl_id": None, "chembl_ids": [],
           "match_method": None, "inchikey": None, "pref_name": None}
    ik = None
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if len(frags) > 1:  # strip counterions -> ChEMBL parent structure
                mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            try:
                ik = Chem.MolToInchiKey(mol)
            except Exception:
                ik = None
    out["inchikey"] = ik

    attempts: list[tuple[dict, str, str]] = []
    if ik:
        attempts.append(({"molecule_structures__standard_inchi_key": ik},
                         "inchikey_exact", f"mol_ik2_{ik}"))
        attempts.append(({"molecule_structures__standard_inchi_key__istartswith": ik[:14]},
                         "inchikey_connectivity", f"mol_ikc2_{ik[:14]}"))
    base = SALT_RE.sub("", name).strip()
    if base:
        attempts.append(({"pref_name__iexact": base}, "pref_name", f"mol_pref2_{base}"))
        attempts.append(({"molecule_synonyms__molecule_synonym__iexact": base},
                         "synonym", f"mol_syn2_{base}"))

    ranked: list[tuple[str, str | None]] = []
    method = None
    for params, meth, key in attempts:
        js = http_json(f"{CHEMBL}/molecule", {**params, "format": "json", "limit": 20},
                       key=key)
        mols = (js or {}).get("molecules") or []
        r = _rank_candidates(mols)
        if r and method is None:
            method = meth
        for c in r:
            if c[0] not in {x[0] for x in ranked}:
                ranked.append(c)
    if ranked:
        out.update(chembl_id=ranked[0][0], chembl_ids=[c for c, _ in ranked[:6]],
                   match_method=method, pref_name=ranked[0][1])
    return out


def fetch_activities(chembl_ids: list[str]) -> list[dict]:
    """Pool qualifying activities over every candidate record for one structure."""
    rows: list[dict] = []
    for chembl_id in chembl_ids:
        for stype in QUAL_TYPES:
            offset, page = 0, 1000
            while True:
                js = http_json(f"{CHEMBL}/activity",
                               {"molecule_chembl_id": chembl_id, "standard_type": stype,
                                "format": "json", "limit": page, "offset": offset},
                               key=f"act_{chembl_id}_{stype}_{offset}")
                acts = (js or {}).get("activities") or []
                for a in acts:
                    rows.append({
                        "molecule_chembl_id": chembl_id,
                        "target_chembl_id": a.get("target_chembl_id"),
                        "target_pref_name": a.get("target_pref_name"),
                        "target_organism": a.get("target_organism"),
                        "standard_type": a.get("standard_type"),
                        "standard_relation": a.get("standard_relation"),
                        "standard_value": a.get("standard_value"),
                        "standard_units": a.get("standard_units"),
                        "pchembl_value": a.get("pchembl_value"),
                        "assay_type": a.get("assay_type"),
                    })
                if len(acts) < page:
                    break
                offset += page
                if offset >= 20000:
                    print(f"    [note] {chembl_id}/{stype} truncated at {offset}",
                          flush=True)
                    break
    return rows


def to_pactivity(row: pd.Series) -> float:
    """pActivity = -log10(molar concentration); prefers curated pchembl_value."""
    p = row.get("pchembl_value")
    if p is not None and not pd.isna(p):
        try:
            return float(p)
        except (TypeError, ValueError):
            pass
    v, u = row.get("standard_value"), row.get("standard_units")
    if v is None or pd.isna(v) or not u:
        return np.nan
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    if v <= 0:
        return np.nan
    scale = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12, "fM": 1e-15}
    if u not in scale:
        return np.nan
    return float(-np.log10(v * scale[u]))


# ---------------------------------------------------------------------------
# Target -> yeast proteome mapping
# ---------------------------------------------------------------------------
def fetch_target_components(target_ids: list[str]) -> pd.DataFrame:
    recs = []
    for i, tid in enumerate(target_ids):
        if i % 50 == 0:
            print(f"    target metadata {i}/{len(target_ids)}", flush=True)
        js = http_json(f"{CHEMBL}/target/{tid}", {"format": "json"}, key=f"tgt_{tid}")
        if not js or js.get("_status") == 404:
            continue
        for c in js.get("target_components") or []:
            syns = c.get("target_component_synonyms") or []
            genes = [s["component_synonym"] for s in syns
                     if s.get("syn_type") in ("GENE_SYMBOL", "GENE_SYMBOL_OTHER")]
            recs.append({"target_chembl_id": tid,
                         "target_type": js.get("target_type"),
                         "organism": js.get("organism"),
                         "accession": c.get("accession"),
                         "component_description": c.get("component_description"),
                         "gene_symbols": "|".join(sorted(set(genes)))})
    return pd.DataFrame(recs)


ACC_RE = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
                    r"|^[A-NR-Z][0-9][A-Z0-9]{3}[0-9][A-Z0-9]{3}[0-9]$")


def _uniprot_batch(chunk: list[str], tag: str) -> pd.DataFrame | None:
    q = " OR ".join(f"accession:{a}" for a in chunk)
    txt = http_text(f"{UNIPROT}/search",
                    {"query": q, "fields": "accession,gene_names,xref_orthodb,organism_id",
                     "format": "tsv", "size": 500},
                    key=f"up_acc_{tag}_{len(chunk)}_{hash(q) & 0xffffffff:08x}",
                    retries=2)
    if txt is None:
        return None
    return pd.read_csv(pd.io.common.StringIO(txt), sep="\t")


def uniprot_table(accessions: list[str]) -> pd.DataFrame:
    """Fetch accession / gene names / OrthoDB group / taxon for accessions.

    A single malformed accession makes UniProt reject the whole batch with
    HTTP 400, which would silently drop ~90 targets from the ortholog mapping.
    On failure we therefore bisect the batch and retry, isolating the offending
    accession instead of losing its neighbours.
    """
    accs = [a for a in accessions if a and ACC_RE.match(str(a))]
    skipped = [a for a in accessions if a and not ACC_RE.match(str(a))]
    if skipped:
        print(f"    [note] {len(skipped)} accession(s) failed the UniProt format "
              f"check and were not queried: {skipped[:5]}", flush=True)
    frames, failed, B = [], [], 90

    def fetch(chunk: list[str], tag: str) -> None:
        if not chunk:
            return
        df = _uniprot_batch(chunk, tag)
        if df is not None:
            frames.append(df)
            return
        if len(chunk) == 1:
            failed.append(chunk[0])
            return
        mid = len(chunk) // 2
        fetch(chunk[:mid], tag + "a")
        fetch(chunk[mid:], tag + "b")

    for i in range(0, len(accs), B):
        fetch(accs[i:i + B], str(i))
        print(f"    uniprot targets {min(i + B, len(accs))}/{len(accs)}", flush=True)
    if failed:
        print(f"    [warn] {len(failed)} accession(s) unresolvable at UniProt: "
              f"{failed[:10]}", flush=True)
    if not frames:
        return pd.DataFrame(columns=["Entry", "Gene Names", "OrthoDB", "Organism (ID)"])
    return pd.concat(frames, ignore_index=True)


def yeast_reference_table() -> pd.DataFrame:
    """Full reviewed S. cerevisiae proteome with gene aliases and OrthoDB groups."""
    txt = http_text(f"{UNIPROT}/stream",
                    {"query": f"organism_id:{YEAST_TAXON} AND reviewed:true",
                     "fields": "accession,gene_names,xref_orthodb", "format": "tsv"},
                    key="up_yeast_proteome")
    if not txt:
        raise SystemExit("Could not download the yeast reference proteome from UniProt.")
    return pd.read_csv(pd.io.common.StringIO(txt), sep="\t")


def _split_groups(s) -> list[str]:
    if not isinstance(s, str):
        return []
    return [g for g in (x.strip() for x in s.split(";")) if g]


def _split_genes(s) -> list[str]:
    if not isinstance(s, str):
        return []
    return [g.strip().upper() for g in s.split() if g.strip()]


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("Step 6.1  ChEMBL target affinity (MoA) vector mapping")
    print("=" * 78, flush=True)

    smiles_map = json.loads((RESULTS / "step4_smiles_resolved.json").read_text())
    names = sorted(smiles_map)
    prot_roster = pd.read_parquet(DATA / "step5_protein_stats.parquet")
    proteins = [str(p) for p in prot_roster["protein"].astype(str)]
    n_prot = len(proteins)
    print(f"[1] roster: {len(names)} perturbations, {n_prot} measured yeast proteins",
          flush=True)

    # ---------------- 2. resolve ----------------
    print("[2] resolving perturbations to ChEMBL molecules ...", flush=True)
    res = []
    for i, nm in enumerate(names):
        if nm in NON_MOA:
            res.append({"name": nm, "smiles": smiles_map[nm].get("smiles"),
                        "chembl_id": None, "chembl_ids": [],
                        "match_method": "excluded_non_moa",
                        "inchikey": None, "pref_name": None})
            continue
        r = resolve_molecule(nm, smiles_map[nm].get("smiles"))
        res.append(r)
        alt = len(r["chembl_ids"])
        print(f"    [{i + 1:>2}/{len(names)}] {nm[:36]:<36} -> "
              f"{r['chembl_id'] or 'NOT FOUND':<15} ({r['match_method']}"
              f"{f', {alt} records pooled' if alt > 1 else ''})", flush=True)
    resolved = pd.DataFrame(res)
    rmap = resolved.set_index("name")
    n_res = int(resolved["chembl_id"].notna().sum())
    print(f"    resolved {n_res}/{len(names)}", flush=True)

    # ---------------- 3. activities ----------------
    print("[3] fetching Kd/Ki/IC50/EC50 activity records ...", flush=True)
    all_acts = []
    sub = resolved[resolved["chembl_id"].notna()].reset_index(drop=True)
    for i, row in sub.iterrows():
        ids = list(row["chembl_ids"]) or [row["chembl_id"]]
        a = fetch_activities(ids)
        for x in a:
            x["name"] = row["name"]
        all_acts.extend(a)
        print(f"    [{i + 1:>2}/{n_res}] {row['name'][:36]:<36} {len(a):>6} records "
              f"over {len(ids)} ChEMBL record(s)", flush=True)
    acts = pd.DataFrame(all_acts)
    if acts.empty:
        raise SystemExit("No activity records retrieved -- cannot build ITPV.")
    acts = acts[acts["target_chembl_id"].notna()].copy()
    acts["pactivity"] = acts.apply(to_pactivity, axis=1)
    n_raw = len(acts)
    acts = acts[np.isfinite(acts["pactivity"])]
    acts = acts[~acts["standard_relation"].isin([">", ">="])].copy()
    print(f"    {n_raw} raw -> {len(acts)} usable, non-censored records", flush=True)
    acts.to_parquet(DATA / "step6_chembl_activities.parquet", index=False)

    agg = (acts.groupby(["name", "target_chembl_id"])
           .agg(pact=("pactivity", "max"), n=("pactivity", "size")).reset_index())
    tgt_ids = sorted(agg["target_chembl_id"].unique().tolist())
    print(f"    {len(tgt_ids)} distinct ChEMBL targets across the panel", flush=True)

    # ---------------- 4. target -> yeast protein ----------------
    print("[4] mapping ChEMBL targets onto the measured yeast proteome ...", flush=True)
    comps = fetch_target_components(tgt_ids)
    comps.to_parquet(DATA / "step6_target_components.parquet", index=False)
    accs = sorted({a for a in comps["accession"].dropna().unique().tolist() if a})
    print(f"    {len(comps)} target components, {len(accs)} distinct UniProt accessions",
          flush=True)

    up_t = uniprot_table(accs)
    up_y = yeast_reference_table()
    print(f"    yeast reference proteome: {len(up_y)} reviewed entries", flush=True)

    # yeast UniProt entry -> measured protein (via any gene alias)
    roster_upper = {}
    for p in proteins:
        roster_upper.setdefault(p.upper(), p)
    y_group_to_prot: dict[str, set[str]] = {}
    y_sym_to_prot: dict[str, set[str]] = {}
    y_acc_to_prot: dict[str, set[str]] = {}
    for _, r in up_y.iterrows():
        genes = _split_genes(r.get("Gene Names"))
        hit = {roster_upper[g] for g in genes if g in roster_upper}
        if not hit:
            continue
        y_acc_to_prot.setdefault(str(r["Entry"]), set()).update(hit)
        for g in genes:
            y_sym_to_prot.setdefault(g, set()).update(hit)
        for grp in _split_groups(r.get("OrthoDB")):
            y_group_to_prot.setdefault(grp, set()).update(hit)
    print(f"    yeast UniProt entries linked to the measured roster: "
          f"{len(y_acc_to_prot)}", flush=True)

    # target accession -> yeast proteins, by channel
    t_up = {}
    for _, r in up_t.iterrows():
        t_up[str(r["Entry"])] = {
            "genes": _split_genes(r.get("Gene Names")),
            "groups": _split_groups(r.get("OrthoDB")),
            "taxon": r.get("Organism (ID)"),
        }
    map_rows = []
    for _, c in comps.iterrows():
        acc = c.get("accession")
        if not acc or acc not in t_up:
            continue
        info = t_up[acc]
        hits: dict[str, str] = {}
        if str(info["taxon"]) == str(YEAST_TAXON) or acc in y_acc_to_prot:
            for p in y_acc_to_prot.get(acc, set()):
                hits.setdefault(p, "direct")
        for grp in info["groups"]:
            for p in y_group_to_prot.get(grp, set()):
                hits.setdefault(p, "orthodb")
        for g in info["genes"] + _split_genes(str(c.get("gene_symbols") or "").replace("|", " ")):
            for p in y_sym_to_prot.get(g, set()):
                hits.setdefault(p, "symbol")
        for p, ch in hits.items():
            map_rows.append({"target_chembl_id": c["target_chembl_id"], "accession": acc,
                             "protein": p, "channel": ch,
                             "target_organism": c.get("organism")})
    tmap = pd.DataFrame(map_rows).drop_duplicates(["target_chembl_id", "protein"])
    tmap.to_parquet(DATA / "step6_target_protein_map.parquet", index=False)
    ch_counts = tmap["channel"].value_counts().to_dict() if len(tmap) else {}
    print(f"    {len(tmap)} target->protein links; by channel: {ch_counts}", flush=True)

    # ---------------- 5. build the 5243-dim ITPV ----------------
    print(f"[5] assembling the {n_prot}-dimensional ITPV ...", flush=True)
    pidx = {p: i for i, p in enumerate(proteins)}
    nidx = {nm: i for i, nm in enumerate(names)}
    ITPV = np.zeros((len(names), n_prot), dtype=np.float32)
    t2p = tmap.groupby("target_chembl_id")["protein"].apply(list).to_dict()
    n_links = 0
    for _, r in agg.iterrows():
        for p in t2p.get(r["target_chembl_id"], ()):
            i, j = nidx[r["name"]], pidx[p]
            if r["pact"] > ITPV[i, j]:
                ITPV[i, j] = r["pact"]
            n_links += 1
    nz_rows = int((ITPV > 0).any(axis=1).sum())
    nz_cols = int((ITPV > 0).any(axis=0).sum())
    density = float((ITPV > 0).mean())
    print(f"    ITPV shape={ITPV.shape}  nonzero density={density:.5f}", flush=True)
    print(f"    {nz_rows}/{len(names)} perturbations and {nz_cols}/{n_prot} proteins "
          f"carry >=1 annotated interaction", flush=True)

    itpv_df = pd.DataFrame(ITPV, index=names, columns=proteins)
    itpv_df.index.name = "chemical"
    itpv_df.reset_index().to_parquet(DATA / "step6_itpv_proteome.parquet", index=False)

    # per-compound ChEMBL-target-space summary (independent of yeast mapping)
    wide = agg.pivot(index="name", columns="target_chembl_id", values="pact").reindex(names)
    has_evidence = wide.notna().any(axis=1)

    # ---------------- 6. compact descriptors ----------------
    print("[6] compact descriptors (SVD fitted on LCGO training compounds only) ...",
          flush=True)
    folds = json.loads((RESULTS / "step5_lcgo_folds.json").read_text())
    train_chems = set(folds["chem_group"].keys())
    train_mask = np.array([nm in train_chems for nm in names])
    print(f"    SVD fit basis: {int(train_mask.sum())} training compounds "
          f"(held-out compounds are transformed, never fitted)", flush=True)

    from sklearn.decomposition import TruncatedSVD
    n_comp = int(min(24, max(2, min(int(train_mask.sum()), nz_cols) - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
    svd.fit(ITPV[train_mask].astype(np.float64))
    Z = svd.transform(ITPV.astype(np.float64))
    evr = float(svd.explained_variance_ratio_.sum())
    print(f"    SVD -> {n_comp} components, cumulative EVR {evr:.4f}", flush=True)

    def _entropy(v: np.ndarray) -> float:
        v = v[v > 0]
        if v.size == 0:
            return 0.0
        w = v / v.sum()
        return float(-(w * np.log(w)).sum())

    feat = pd.DataFrame(index=pd.Index(names, name="chemical"))
    feat["has_chembl_evidence"] = has_evidence.values.astype(float)
    feat["itpv_n_chembl_targets"] = wide.notna().sum(axis=1).values.astype(float)
    feat["itpv_max_pactivity"] = np.nan_to_num(wide.max(axis=1).values)
    feat["itpv_mean_pactivity"] = np.nan_to_num(wide.mean(axis=1).values)
    feat["itpv_n_yeast_targets"] = (ITPV > 0).sum(axis=1).astype(float)
    feat["itpv_yeast_max_pactivity"] = ITPV.max(axis=1)
    feat["itpv_yeast_sum_pactivity"] = ITPV.sum(axis=1)
    feat["itpv_selectivity_entropy"] = [_entropy(ITPV[i]) for i in range(len(names))]
    for j in range(n_comp):
        feat[f"itpv_svd{j:02d}"] = Z[:, j]

    assert not feat.isna().any().any(), "compact ITPV features contain NaN"
    feat.reset_index().to_parquet(DATA / "step6_target_features.parquet", index=False)
    print(f"    saved data/step6_target_features.parquet shape={feat.shape}", flush=True)

    # ---------------- 7. report ----------------
    per = []
    for nm in names:
        per.append({"chemical": nm,
                    "chembl_id": rmap.loc[nm, "chembl_id"],
                    "match_method": rmap.loc[nm, "match_method"],
                    "n_chembl_targets": int(wide.loc[nm].notna().sum()),
                    "n_yeast_targets": int((ITPV[nidx[nm]] > 0).sum()),
                    "max_pactivity": float(ITPV[nidx[nm]].max()),
                    "has_evidence": bool(has_evidence.loc[nm])})
    report = {
        "step": "6_1_target_affinity_features",
        "seed": SEED,
        "n_perturbations": len(names),
        "n_resolved_to_chembl": n_res,
        "n_with_activity_evidence": int(has_evidence.sum()),
        "n_activity_records_raw": int(n_raw),
        "n_activity_records_usable": int(len(acts)),
        "qualifying_activity_types": list(QUAL_TYPES),
        "n_distinct_chembl_targets": len(tgt_ids),
        "n_target_uniprot_accessions": len(accs),
        "itpv_dimension": n_prot,
        "itpv_dimension_matches_brief_5243": bool(n_prot == 5243),
        "itpv_dimension_note": (
            "The ITPV is defined over the measured yeast proteome, so its width is "
            f"{n_prot} -- exactly the 5,243 proteins quantified in this study. Each "
            "entry is the strongest annotated binding potency (pActivity = -log10[M]) "
            "of that compound against that protein, obtained by mapping ChEMBL "
            "targets onto the yeast proteome."),
        "target_to_protein_links": int(len(tmap)),
        "mapping_channels": {str(k): int(v) for k, v in ch_counts.items()},
        "mapping_channel_definitions": {
            "direct": "ChEMBL target is itself an S. cerevisiae protein in the roster",
            "orthodb": "target and yeast protein share an OrthoDB Eukaryota orthologous group",
            "symbol": "target and yeast protein share a gene symbol or alias",
        },
        "itpv_nonzero_density": density,
        "n_perturbations_with_nonzero_itpv": nz_rows,
        "n_proteins_with_nonzero_itpv": nz_cols,
        "sparsity_note": (
            "The ITPV is sparse by construction: only a minority of the 5,243 measured "
            "proteins are annotated drug targets in ChEMBL. Coverage is reported as "
            "measured. A zero entry means 'no annotated engagement', which is a "
            "substantive statement, not a missing value -- no imputation is performed."),
        "missing_value_policy": (
            "Perturbations with no ChEMBL activity data at all (solvents, inorganic "
            "stressors, vehicle controls) carry an all-zero ITPV together with "
            "has_chembl_evidence = 0, so models can distinguish 'no annotated target' "
            "from 'not looked up'. The compact feature matrix contains zero NaNs."),
        "n_nan_in_feature_matrix": int(feat.isna().sum().sum()),
        "compact_feature_shape": list(feat.shape),
        "svd_components": n_comp,
        "svd_fit_basis": f"{int(train_mask.sum())} LCGO training compounds only",
        "svd_explained_variance_ratio_sum": evr,
        "top_target_organisms": {str(k): int(v) for k, v in
                                 acts.groupby("target_organism").size()
                                 .sort_values(ascending=False).head(15).items()},
        "per_compound": per,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (RESULTS / "step6_target_affinity_report.json").write_text(json.dumps(report, indent=2))
    print("[7] -> results/step6_target_affinity_report.json", flush=True)
    print(f"DONE in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
