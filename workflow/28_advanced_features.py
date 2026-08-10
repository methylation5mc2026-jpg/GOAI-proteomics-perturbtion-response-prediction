"""Step 5.2 -- 3D molecular features and STRING protein-graph embeddings.

Two independent feature blocks, run separately because they sit on different
critical paths:

``--part mol3d``
    Conformer-dependent chemistry for the 57 perturbation labels. Step 4 gave the
    GBDTs a purely topological (2D) view: descriptors, ECFP4 bits and their PCA.
    A 2D fingerprint cannot distinguish a flat, rigid, planar intercalator from a
    globular flexible macrolide of the same composition, yet those two things
    reach very different intracellular compartments. So this part embeds one
    ETKDGv3 conformer per compound, optimises it with MMFF94 where the atom types
    are covered, and reads off shape and surface descriptors that only exist in
    3D: Free-SASA, principal moments of inertia, the normalised PMI ratios
    NPR1/NPR2, asphericity, eccentricity, spherocity, inertial shape factor,
    radius of gyration, plane-of-best-fit deviation, and a Gobbi **3D**
    pharmacophore fingerprint (pharmacophore pairs binned by through-space
    distance rather than bond count).

    Conformer generation is stochastic, so the seed is fixed and the number of
    embedding attempts is logged per compound; a compound whose embedding fails
    keeps the explicit zero vector and the ``mol3d_has_conformer = 0`` flag,
    exactly the convention Step 4 used for the non-molecular ``Quality Control``
    label. Nothing is silently imputed.

``--part graph``
    A STRING v12 protein-protein interaction graph over the 5,243 measured yeast
    proteins, reduced to 64 spectral embedding dimensions, and a K-means
    partition of the protein axis used later as the cluster index of the
    stacking weight tensor.

    Why a graph at all: the stacking meta-learner needs to know which proteins
    behave alike, and the honest options are (a) topology from an external
    interaction database or (b) co-response structure measured in the training
    rows. Both are used -- the embedding is concatenated from STRING spectral
    coordinates *and* train-only abundance/response statistics -- because the
    cluster index has to separate hyper-abundant enzymes from low-abundance
    regulators (an abundance property) as well as co-regulated modules (a
    topology property).

Leakage discipline
------------------
* the 3D pharmacophore PCA is fitted on **train compounds only**, mirroring
  ``22_rdkit_features.py``;
* protein clusters are fitted on **train rows only** -- no val or test row
  contributes to a cluster centroid, so the cluster index transfers to the test
  set the same way the regime routing does.

Provenance
----------
STRING identifiers, edges and their confidence scores are downloaded live and
the exact query URLs plus the resolved database version are recorded in
``results/step5_graph_report.json``. If STRING is unreachable the script falls
back to a train-only co-response graph and **records which graph was used**, so
a downstream reader can never mistake one for the other.

Outputs
-------
data/step5_mol3d_features.parquet    per-compound 3D block
data/step5_protein_graph.npz         adjacency (sparse), spectral embedding
data/step5_protein_clusters.parquet  protein -> cluster id, per K
results/step5_mol3d_report.json      conformer QC
results/step5_graph_report.json      STRING mapping/edge QC + provenance
figures/step5_knowledge_graph_embedding.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_common as S4  # noqa: E402

DATA, RESULTS, FIGURES, WORKFLOW = S4.DATA, S4.RESULTS, S4.FIGURES, S4.WORKFLOW
SEED, CHEM_COL, log = S4.SEED, S4.CHEM_COL, S4.log

STRING_SPECIES = 4932  # Saccharomyces cerevisiae S288c
STRING_MIN_SCORE = 400  # STRING's own "medium confidence" cut-off
STRING_FILE_VERSION = "12.0"  # bulk protein.links release used for the whole-proteome graph
N_SPECTRAL = 64
CLUSTER_KS = (1, 4, 8, 12, 16)

#: One protein label in the released matrix is an Excel-mangled gene symbol:
#: '1-Oct' is what a spreadsheet does to OCT1. Recorded and repaired explicitly
#: rather than dropped, because a silently unmapped protein would look like a
#: STRING coverage failure instead of a data-provenance artefact.
ID_REPAIRS = {"1-Oct": "OCT1"}


# ===========================================================================
# Part 1 -- 3D molecular features
# ===========================================================================
def build_mol3d(n_conf: int = 8, pharm_pca: int = 12) -> None:
    """Embed one optimised conformer per compound and read off 3D descriptors.

    Parameters
    ----------
    n_conf : int
        Conformers generated per compound; the lowest-MMFF-energy one is kept
        (falling back to the first embedded conformer when MMFF cannot type the
        molecule, e.g. the platinum coordination centre of cisplatin).
    pharm_pca : int
        Components retained from the PCA of the 3D pharmacophore fingerprint,
        fitted on train compounds only.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors3D, rdFreeSASA, rdMolDescriptors
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

    RDLogger.DisableLog("rdApp.*")

    smi_map = json.loads(
        (RESULTS / "step4_smiles_resolved.json").read_text(encoding="utf-8")
    )
    tr = pd.read_csv(DATA / "meta_train_val_annotated.csv")
    train_names = set(tr[CHEM_COL].dropna().astype(str))
    names = sorted(smi_map)
    log(f"3D featurisation of {len(names)} labels ({len(train_names & set(names))} seen in train)")

    #: 3D descriptors that are pure functions of the heavy-atom coordinates.
    d3_fns = {
        "PMI1": Descriptors3D.PMI1,
        "PMI2": Descriptors3D.PMI2,
        "PMI3": Descriptors3D.PMI3,
        "NPR1": Descriptors3D.NPR1,
        "NPR2": Descriptors3D.NPR2,
        "Asphericity": Descriptors3D.Asphericity,
        "Eccentricity": Descriptors3D.Eccentricity,
        "InertialShapeFactor": Descriptors3D.InertialShapeFactor,
        "RadiusOfGyration": Descriptors3D.RadiusOfGyration,
        "SpherocityIndex": Descriptors3D.SpherocityIndex,
        "PBF": rdMolDescriptors.CalcPBF,
    }

    rows: list[dict] = []
    pharm_bits: list[np.ndarray] = []
    audit: list[dict] = []
    n_pharm_bits = None

    t0 = time.time()
    for i, name in enumerate(names, 1):
        rec = smi_map[name]
        smi = rec.get("smiles")
        ent = {"name": name, "cid": rec.get("cid"), "smiles": smi}
        row = {"name": name, "mol3d_has_conformer": 0.0}

        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            ent.update(status="no_structure")
            rows.append(row)
            pharm_bits.append(None)
            audit.append(ent)
            log(f"  [{i}/{len(names)}] {name!r}: no structure -> zero 3D vector")
            continue

        # Same conservative salt handling as Step 4: keep the largest fragment
        # when it is unambiguously the parent, otherwise keep the whole species.
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            uniq = {}
            for f in frags:
                uniq.setdefault(Chem.MolToSmiles(f), f)
            cands = list(uniq.values())
            sizes = [f.GetNumHeavyAtoms() for f in cands]
            b = int(np.argmax(sizes))
            if sizes[b] >= 6 and sizes[b] > sum(sizes) - sizes[b]:
                mol = cands[b]
                ent["fragment_decision"] = f"stripped_to_parent({len(frags)}_frags)"
            else:
                ent["fragment_decision"] = "kept_intact_multi_fragment"

        molh = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = SEED
        ps.useSmallRingTorsions = True
        cids = AllChem.EmbedMultipleConfs(molh, numConfs=n_conf, params=ps)
        ent["n_conformers_embedded"] = len(cids)

        if not len(cids):
            # Random-coordinate fallback: coordination compounds (cisplatin) have
            # no ETKDG torsion model, but their geometry is still well defined.
            ps2 = AllChem.ETKDGv3()
            ps2.randomSeed = SEED
            ps2.useRandomCoords = True
            cids = AllChem.EmbedMultipleConfs(molh, numConfs=n_conf, params=ps2)
            ent["n_conformers_embedded"] = len(cids)
            ent["embedding_fallback"] = "useRandomCoords"

        if not len(cids):
            ent["status"] = "embedding_failed"
            rows.append(row)
            pharm_bits.append(None)
            audit.append(ent)
            log(f"  [{i}/{len(names)}] {name!r}: conformer embedding FAILED -> zero 3D vector")
            continue

        best_cid = int(cids[0])
        try:
            res = AllChem.MMFFOptimizeMoleculeConfs(molh, maxIters=800)
            energies = [e for _, e in res]
            if all(np.isfinite(energies)):
                best_cid = int(cids[int(np.argmin(energies))])
                ent["mmff"] = "optimised"
                ent["mmff_energy"] = float(min(energies))
                ent["mmff_not_converged"] = int(sum(c for c, _ in res))
            else:
                ent["mmff"] = "non_finite_energy_kept_first_conformer"
        except Exception as exc:
            # MMFF cannot type every element (Pt in cisplatin). The unoptimised
            # ETKDG geometry is still a legitimate conformer, so it is kept and
            # the failure is recorded rather than swallowed.
            ent["mmff"] = f"unavailable ({type(exc).__name__})"

        conf_mol = Chem.Mol(molh, False, best_cid) if len(cids) > 1 else molh
        try:
            conf_mol = Chem.Mol(molh)
            keep = conf_mol.GetConformer(best_cid)
            conf_mol.RemoveAllConformers()
            conf_mol.AddConformer(keep, assignId=True)
        except Exception:
            conf_mol = molh

        row["mol3d_has_conformer"] = 1.0
        for k, fn in d3_fns.items():
            try:
                row[f"mol3d_{k}"] = float(fn(conf_mol))
            except Exception:
                row[f"mol3d_{k}"] = np.nan

        # Free-SASA needs a radius for every element. Pt (cisplatin) and the
        # bare ions of NaCl are not in the classifier's typing table, so SASA is
        # genuinely *undefined* for them rather than zero -- an inorganic salt
        # obviously has a surface. A separate flag carries that distinction, so a
        # tree splitting on SASA sees "unavailable", not "no surface area".
        try:
            radii = rdFreeSASA.classifyAtoms(conf_mol)
            row["mol3d_SASA"] = float(rdFreeSASA.CalcSASA(conf_mol, radii))
            n_heavy = conf_mol.GetNumHeavyAtoms()
            row["mol3d_SASA_per_heavy_atom"] = row["mol3d_SASA"] / max(n_heavy, 1)
            row["mol3d_sasa_defined"] = 1.0
        except Exception as exc:
            row["mol3d_SASA"] = np.nan
            row["mol3d_SASA_per_heavy_atom"] = np.nan
            row["mol3d_sasa_defined"] = 0.0
            ent["sasa"] = f"undefined ({type(exc).__name__}: element not in the radius table)"

        # Through-space pharmacophore pairs: same Gobbi feature definitions as
        # the 2D fingerprint, but binned by 3D distance, so it encodes geometry
        # the ECFP4 block structurally cannot see.
        try:
            dmat = Chem.Get3DDistanceMatrix(conf_mol)
            fp3 = Generate.Gen2DFingerprint(conf_mol, Gobbi_Pharm2D.factory, dMat=dmat)
            v = np.zeros(fp3.GetNumBits(), dtype=np.uint8)
            for b in fp3.GetOnBits():
                v[b] = 1
            pharm_bits.append(v)
            n_pharm_bits = len(v)
            ent["n_pharm3d_on_bits"] = int(v.sum())
        except Exception as exc:
            pharm_bits.append(None)
            ent["pharm3d"] = f"failed ({type(exc).__name__})"

        ent["status"] = "ok"
        rows.append(row)
        audit.append(ent)
        if i % 5 == 0 or i == len(names):
            log(f"  [{i}/{len(names)}] 3D descriptors done | {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows).set_index("name")
    for flag in ("mol3d_has_conformer", "mol3d_sasa_defined"):
        if flag not in df.columns:
            df[flag] = 0.0
        df[flag] = df[flag].fillna(0.0)
    d3_cols = [c for c in df.columns
               if c not in ("mol3d_has_conformer", "mol3d_sasa_defined")]
    n_missing_cells = int(df[d3_cols].isna().sum().sum())
    # Compounds without a conformer keep an explicit zero vector, flagged by
    # mol3d_has_conformer = 0 -- the same convention Step 4 used, so a tree can
    # branch on "no 3D information" instead of on a fabricated value.
    df[d3_cols] = df[d3_cols].fillna(0.0)
    log(f"  filled {n_missing_cells} missing 3D descriptor cells with the flagged zero vector")

    # ---- pharmacophore fingerprint -> train-fitted PCA --------------------
    if n_pharm_bits is None:
        raise RuntimeError("no 3D pharmacophore fingerprint could be generated at all")
    P = np.zeros((len(df), n_pharm_bits), dtype=np.float32)
    for i, v in enumerate(pharm_bits):
        if v is not None:
            P[i] = v
    tr_idx = [i for i, n in enumerate(df.index) if n in train_names]
    on_tr = P[tr_idx].sum(axis=0)
    keep = np.where((on_tr >= 2) & (on_tr <= len(tr_idx) - 2))[0]
    log(f"  3D pharmacophore: {n_pharm_bits} bits -> {len(keep)} informative (train-only filter)")

    from sklearn.decomposition import PCA

    n_comp = int(min(pharm_pca, max(len(tr_idx) - 1, 1), max(len(keep), 1)))
    pca = PCA(n_components=n_comp, random_state=SEED)
    pca.fit(P[tr_idx][:, keep].astype(np.float64))
    sc = pca.transform(P[:, keep].astype(np.float64))
    evr = pca.explained_variance_ratio_
    log(f"  3D pharmacophore PCA: {n_comp} comps, cumulative EVR = {evr.sum():.3f}")
    pdf = pd.DataFrame(
        sc.astype("float32"),
        index=df.index,
        columns=[f"ph3pca_{i:02d}" for i in range(n_comp)],
    )

    out = pd.concat([df.astype("float32"), pdf], axis=1)
    out.index.name = CHEM_COL
    p = DATA / "step5_mol3d_features.parquet"
    out.to_parquet(p, compression="snappy")
    log(f"  wrote {p} {out.shape}")

    n_conf_ok = int(out["mol3d_has_conformer"].sum())
    S4.write_json(
        RESULTS / "step5_mol3d_report.json",
        {
            "step": "5_2_mol3d",
            "seed": SEED,
            "n_labels": len(out),
            "n_with_conformer": n_conf_ok,
            "n_without_conformer": int(len(out) - n_conf_ok),
            "n_train_compounds": len(tr_idx),
            "n_conformers_requested_per_compound": n_conf,
            "n_missing_descriptor_cells_filled_with_flagged_zero": n_missing_cells,
            "pharm3d": {
                "n_raw_bits": int(n_pharm_bits),
                "n_informative_bits_train_filter": int(len(keep)),
                "n_pca_components": int(n_comp),
                "cumulative_explained_variance_ratio": float(evr.sum()),
                "explained_variance_ratio": [round(float(x), 5) for x in evr],
            },
            "columns": list(out.columns),
            "leakage_note": (
                "the informative-bit filter and the PCA are fitted on train compounds only; "
                "test-only compounds are transformed, never fitted"
            ),
            "per_compound_audit": audit,
        },
    )


# ===========================================================================
# Part 2 -- STRING PPI graph, spectral embedding, protein clusters
# ===========================================================================
def _string_map_ids(proteins: list[str], report: dict) -> pd.DataFrame:
    """Resolve gene symbols to STRING identifiers in batches.

    STRING's ``get_string_ids`` endpoint accepts a few hundred identifiers per
    call. ``echo_query=1`` is requested so the mapping can be joined on the
    submitted string rather than on a positional index, which would silently
    misalign whenever STRING drops an unmappable entry.
    """
    import requests

    url = "https://string-db.org/api/tsv/get_string_ids"
    report["queries"] = report.get("queries", [])
    out: list[dict] = []
    B = 400
    for s in range(0, len(proteins), B):
        chunk = proteins[s : s + B]
        r = requests.post(
            url,
            data={
                "identifiers": "\r".join(chunk),
                "species": STRING_SPECIES,
                "limit": 1,
                "echo_query": 1,
                "caller_identity": "goai-virtual-cell_step5",
            },
            timeout=180,
        )
        r.raise_for_status()
        lines = r.text.strip().splitlines()
        hdr = lines[0].split("\t")
        for ln in lines[1:]:
            f = ln.split("\t")
            if len(f) == len(hdr):
                out.append(dict(zip(hdr, f)))
        report["queries"].append({"url": url, "n_identifiers": len(chunk), "status": r.status_code})
        log(f"    mapped {min(s + B, len(proteins))}/{len(proteins)} identifiers")
    return pd.DataFrame(out)


def _string_network(report: dict) -> pd.DataFrame:
    """Download the complete STRING interaction file for the species.

    The REST ``network`` endpoint only returns edges **among the identifiers
    submitted in that call**, so requesting 5,185 proteins in batches yields only
    intra-batch edges. That failure is silent and looks like a sparse network: an
    earlier attempt here batched at 900 identifiers and produced 41,936 edges in
    170 components whose five largest were ~880 nodes each -- exactly the batch
    size, which is the tell. The bulk ``protein.links`` file is the correct source
    for a whole-proteome graph and is what is used.
    """
    import gzip
    import io

    import requests

    url = (
        f"https://stringdb-downloads.org/download/protein.links.v{STRING_FILE_VERSION}/"
        f"{STRING_SPECIES}.protein.links.v{STRING_FILE_VERSION}.txt.gz"
    )
    log(f"    downloading the full species link file: {url}")
    r = requests.get(url, timeout=900)
    r.raise_for_status()
    raw = gzip.decompress(r.content)
    df = pd.read_csv(io.BytesIO(raw), sep=" ")
    report["queries"].append(
        {"url": url, "status": r.status_code, "n_bytes_gz": len(r.content),
         "n_edge_rows_in_file": int(len(df)),
         "columns": list(df.columns)}
    )
    log(f"    link file: {len(df):,} scored edges, columns {list(df.columns)}")
    score_col = "combined_score" if "combined_score" in df.columns else df.columns[-1]
    before = len(df)
    df = df[df[score_col] >= STRING_MIN_SCORE]
    log(f"    {len(df):,} edges at combined_score >= {STRING_MIN_SCORE} "
        f"(dropped {before - len(df):,})")
    report["n_edges_in_species_file"] = int(before)
    report["n_edges_above_score_cutoff"] = int(len(df))
    return df.rename(columns={df.columns[0]: "stringId_A",
                              df.columns[1]: "stringId_B",
                              score_col: "score"})


def build_graph(k_spectral: int = N_SPECTRAL) -> None:
    """Build the protein interaction graph, its embedding and the clusters."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    sys.path.insert(0, str(WORKFLOW))
    import validation_splits as VS

    log("loading the train_val cohort for train-only protein statistics ...")
    meta, Y, D, C, proteins = VS.load_eval_data()
    masks = VS.split_masks(meta)
    train_mask = masks[VS.TRAIN_SPLIT]
    n_p = len(proteins)
    log(f"  {Y.shape[0]} samples x {n_p} proteins; train rows = {int(train_mask.sum())}")

    report: dict = {
        "step": "5_2_graph",
        "seed": SEED,
        "species_taxid": STRING_SPECIES,
        "required_score": STRING_MIN_SCORE,
        "n_proteins": n_p,
        "id_repairs": ID_REPAIRS,
    }

    # ---- STRING mapping and edges ----------------------------------------
    query_names = [ID_REPAIRS.get(p, p) for p in proteins]
    adj = None
    graph_source = "none"
    try:
        import requests

        v = requests.get("https://string-db.org/api/tsv/version", timeout=60)
        v.raise_for_status()
        report["string_version"] = v.text.strip().splitlines()[-1]
        log(f"STRING version: {report['string_version']}")

        log("resolving protein identifiers against STRING ...")
        mp = _string_map_ids(query_names, report)
        qcol = "queryItem" if "queryItem" in mp.columns else "queryIndex"
        mp = mp.drop_duplicates(subset=[qcol])
        sid_of = dict(zip(mp[qcol].astype(str), mp["stringId"].astype(str)))
        name_of_string: dict[str, list[int]] = {}
        mapped = 0
        for i, q in enumerate(query_names):
            sid = sid_of.get(q)
            if sid:
                name_of_string.setdefault(sid, []).append(i)
                mapped += 1
        report["n_mapped_to_string"] = mapped
        report["n_unmapped"] = int(n_p - mapped)
        report["unmapped_examples"] = [
            proteins[i] for i, q in enumerate(query_names) if q not in sid_of
        ][:40]
        log(f"  mapped {mapped}/{n_p} proteins ({100 * mapped / n_p:.1f}%); "
            f"{n_p - mapped} unmapped")

        log("downloading the STRING interaction network ...")
        net = _string_network(report)
        report["n_edge_rows_downloaded"] = int(len(net))
        if len(net):
            # The bulk file is keyed by STRING identifier, which is exactly what
            # get_string_ids returned, so the join is on a stable primary key --
            # no name normalisation and no chance of a positional misalignment.
            # A STRING id may map to more than one matrix column if two measured
            # labels resolve to the same protein; every such pair gets the edge.
            sid_to_idx: dict[str, list[int]] = {}
            for sid, idxs in name_of_string.items():
                sid_to_idx.setdefault(sid, []).extend(idxs)

            a_idx = net["stringId_A"].astype(str).map(sid_to_idx)
            b_idx = net["stringId_B"].astype(str).map(sid_to_idx)
            keep = a_idx.notna() & b_idx.notna()
            report["n_edge_rows_unresolved"] = int((~keep).sum())
            log(f"    {int(keep.sum()):,} edges have both endpoints among the measured "
                f"proteins ({int((~keep).sum()):,} dropped)")

            rows, cols, vals = [], [], []
            for ia, ib, w in zip(
                a_idx[keep].to_numpy(), b_idx[keep].to_numpy(),
                net.loc[keep, "score"].astype(float).to_numpy(),
            ):
                for x in ia:
                    for y in ib:
                        if x != y:
                            rows.append(x)
                            cols.append(y)
                            vals.append(w)
            A = sp.coo_matrix((vals, (rows, cols)), shape=(n_p, n_p)).tocsr()
            A = A.maximum(A.T)  # STRING scores are symmetric
            A.setdiag(0.0)
            A.eliminate_zeros()
            adj = A
            graph_source = f"STRING_v{STRING_FILE_VERSION}_PPI_bulk_links"
            deg = np.asarray(A.sum(axis=1)).ravel()
            report["n_edges_undirected"] = int(A.nnz // 2)
            report["n_isolated_nodes"] = int((deg == 0).sum())
            report["mean_weighted_degree"] = float(deg.mean())
            log(f"  STRING graph: {A.nnz // 2:,} undirected edges, "
                f"{int((deg == 0).sum())} isolated nodes")
    except Exception as exc:
        report["string_error"] = f"{type(exc).__name__}: {exc}"
        log(f"  !! STRING unavailable: {type(exc).__name__}: {exc}")

    # ---- fallback / complement: train-only co-response graph --------------
    # Built regardless, because it is also a diagnostic: if the STRING graph and
    # the measured co-response graph disagree completely, clustering on STRING
    # alone would be clustering on something the data does not express.
    log("building the train-only co-response graph (kNN on Delta correlation) ...")
    Dtr = D[train_mask]
    with np.errstate(all="ignore"):
        Dz = Dtr - np.nanmean(Dtr, axis=0, keepdims=True)
        sd = np.nanstd(Dz, axis=0, keepdims=True)
        Dz = np.where(np.isfinite(Dz), Dz, 0.0) / np.where(sd > 0, sd, 1.0)
    n_obs = np.isfinite(Dtr).sum(axis=0)
    Dz[:, n_obs < 20] = 0.0
    Ktop = 20
    Rcorr = (Dz.T @ Dz) / max(len(Dz) - 1, 1)
    np.fill_diagonal(Rcorr, -np.inf)
    nn = np.argpartition(-Rcorr, Ktop, axis=1)[:, :Ktop]
    rows = np.repeat(np.arange(n_p), Ktop)
    cols = nn.ravel()
    vals = np.clip(Rcorr[rows, cols], 0.0, None)
    Aco = sp.coo_matrix((vals, (rows, cols)), shape=(n_p, n_p)).tocsr()
    Aco = Aco.maximum(Aco.T)
    Aco.setdiag(0.0)
    Aco.eliminate_zeros()
    del Rcorr
    log(f"  co-response graph: {Aco.nnz // 2} undirected edges (k={Ktop})")
    report["coresponse_graph"] = {
        "k_neighbours": Ktop,
        "n_edges_undirected": int(Aco.nnz // 2),
        "n_proteins_with_too_few_observations": int((n_obs < 20).sum()),
        "note": "Pearson correlation of train-row Delta, NaN treated as 0 after standardisation",
    }

    if adj is None:
        adj = Aco
        graph_source = "train_only_coresponse_kNN_FALLBACK"
        log("  !! using the co-response graph as the primary graph (STRING failed)")
    else:
        # Agreement diagnostic between the two independent graphs.
        inter = adj.astype(bool).multiply(Aco.astype(bool)).nnz // 2
        report["string_vs_coresponse_shared_edges"] = int(inter)
        report["string_vs_coresponse_jaccard"] = float(
            inter / max((adj.nnz + Aco.nnz) // 2 - inter, 1)
        )
        log(f"  STRING and co-response graphs share {inter} edges "
            f"(Jaccard {report['string_vs_coresponse_jaccard']:.4f})")

    report["graph_source"] = graph_source

    # ---- spectral embedding ----------------------------------------------
    # The graph is NOT connected: the normalised adjacency has one eigenvalue of
    # exactly 1 per connected component, and those eigenvectors are component
    # indicators, not community structure. Feeding them to K-means spends whole
    # clusters on 2-protein fragments (measured: K=8 produced clusters of size
    # 2, 2 and 9). So the embedding is computed on the giant component only, the
    # single trivial eigenvector there is dropped, and every off-giant protein
    # gets an explicit zero embedding plus a flag.
    from scipy.sparse.csgraph import connected_components

    deg = np.asarray(adj.sum(axis=1)).ravel()
    n_comp, comp_id = connected_components(adj, directed=False)
    sizes = np.bincount(comp_id)
    giant = int(np.argmax(sizes))
    in_giant = comp_id == giant
    log(f"  graph has {n_comp} connected components; giant component holds "
        f"{int(in_giant.sum())}/{n_p} proteins "
        f"(next largest {sorted(sizes.tolist(), reverse=True)[1:6]})")

    log(f"computing the {k_spectral}-dimensional normalised-Laplacian embedding "
        f"on the giant component ...")
    gi = np.flatnonzero(in_giant)
    Ag = adj[gi][:, gi]
    dg = np.asarray(Ag.sum(axis=1)).ravel()
    dginv = 1.0 / np.sqrt(np.where(dg > 0, dg, 1.0))
    An = (sp.diags(dginv) @ Ag @ sp.diags(dginv)).astype(np.float64)
    t0 = time.time()
    kk = int(min(k_spectral + 1, len(gi) - 2))
    evals_g, evecs_g = eigsh(An, k=kk, which="LA", tol=1e-6, maxiter=8000)
    order = np.argsort(-evals_g)
    evals_g, evecs_g = evals_g[order], evecs_g[:, order]
    # Drop the trivial top eigenvector (eigenvalue 1, proportional to sqrt(deg)):
    # it carries no community information on a connected graph.
    evals_k, evecs_k = evals_g[1:], evecs_g[:, 1:]
    log(f"  eigsh converged in {time.time() - t0:.0f}s; trivial lambda={evals_g[0]:.6f} "
        f"dropped, kept lambda in [{evals_k[-1]:.4f}, {evals_k[0]:.4f}]")

    emb = np.zeros((n_p, evecs_k.shape[1]), dtype="float32")
    emb[gi] = (evecs_k * dginv[:, None]).astype("float32")
    evals = evals_k
    report["spectral"] = {
        "k_kept": int(evecs_k.shape[1]),
        "n_connected_components": int(n_comp),
        "giant_component_size": int(in_giant.sum()),
        "component_sizes_top10": sorted(sizes.tolist(), reverse=True)[:10],
        "trivial_eigenvalue_dropped": float(evals_g[0]),
        "eigenvalues_top10_after_dropping_trivial": [float(x) for x in evals_k[:10]],
        "n_off_giant_zero_embedding": int((~in_giant).sum()),
        "n_isolated_nodes": int((deg == 0).sum()),
        "note": (
            "one eigenvalue of the normalised adjacency equals 1 per connected component, so "
            "those eigenvectors are component indicators rather than community structure; the "
            "embedding is therefore restricted to the giant component and the trivial "
            "eigenvector is dropped"
        ),
    }

    # ---- train-only protein statistics -----------------------------------
    log("computing train-only per-protein abundance / response statistics ...")
    with np.errstate(all="ignore"):
        Ytr = Y[train_mask]
        stat = pd.DataFrame(
            {
                "protein": proteins,
                "abundance_mean": np.nanmean(Ytr, axis=0),
                "abundance_sd": np.nanstd(Ytr, axis=0),
                "detect_rate": np.isfinite(Ytr).mean(axis=0),
                "delta_sd": np.nanstd(Dtr, axis=0),
                "delta_abs_mean": np.nanmean(np.abs(Dtr), axis=0),
                "weighted_degree": deg,
                "in_giant_component": in_giant.astype(np.float64),
                "component_id": comp_id.astype(np.int64),
            }
        )
    med = stat.median(numeric_only=True)
    n_na = int(stat.isna().sum().sum())
    stat = stat.fillna(med).fillna(0.0)
    log(f"  {n_na} statistic cells were undefined (never-detected proteins) -> train median")

    # ---- clustering -------------------------------------------------------
    # The cluster index must separate BOTH co-regulated modules (a topology
    # property, from the embedding) AND abundance strata (a scale property, from
    # the statistics), because the rubric's R^2 terms are scale-sensitive.
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import QuantileTransformer

    stat_cols = ["abundance_mean", "abundance_sd", "detect_rate", "delta_sd",
                 "delta_abs_mean", "weighted_degree"]
    # Rank-based (quantile) scaling rather than z-scoring. Spectral coordinates of
    # a sparse graph are heavy-tailed -- a handful of proteins in a small dense
    # neighbourhood carry enormous z-scores -- and weighted degree is similarly
    # skewed. Under z-scoring K-means spends whole clusters on those few
    # extreme proteins; a rank transform bounds every coordinate identically.
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=SEED, subsample=100_000)
    Xs = qt.fit_transform(stat[stat_cols].to_numpy("float64"))
    n_sp = int(min(16, emb.shape[1]))
    Es = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=SEED, subsample=100_000
                             ).fit_transform(emb[:, :n_sp].astype("float64"))
    # Equal total variance from each view so neither dominates by dimension count.
    Z = np.hstack([Xs / np.sqrt(Xs.shape[1]), Es / np.sqrt(Es.shape[1])])
    log(f"  clustering feature matrix {Z.shape} (abundance stats + {n_sp} spectral coords, "
        f"rank-transformed)")

    def enforce_min_size(lab: np.ndarray, K: int, min_size: int) -> tuple[np.ndarray, list]:
        """Merge undersized clusters into their nearest surviving centroid.

        A cluster with a handful of proteins would carry its own row of the
        stacking weight tensor while contributing almost nothing to the objective,
        so its weights would be fitted on noise. Merging is reported, not silent.
        """
        merges = []
        lab = lab.copy()
        while True:
            present = np.unique(lab)
            if len(present) <= 1:
                break
            sizes = {int(c): int((lab == c).sum()) for c in present}
            small = [c for c, n in sizes.items() if n < min_size]
            if not small:
                break
            victim = min(small, key=lambda c: sizes[c])
            cents = {int(c): Z[lab == c].mean(axis=0) for c in present if c != victim}
            vc = Z[lab == victim].mean(axis=0)
            target = min(cents, key=lambda c: float(np.sum((cents[c] - vc) ** 2)))
            merges.append({"merged_cluster_size": sizes[victim], "into_cluster_of_size":
                           sizes[target]})
            lab[lab == victim] = target
        # compact the labels to 0..k-1
        uniq = {int(c): i for i, c in enumerate(np.unique(lab))}
        return np.array([uniq[int(c)] for c in lab], dtype=int), merges

    clus = pd.DataFrame({"protein": proteins})
    prof: dict = {}
    min_cluster = max(50, n_p // (4 * max(max(CLUSTER_KS), 1)))
    log(f"  minimum cluster size enforced: {min_cluster} proteins")
    for K_req in CLUSTER_KS:
        merges: list = []
        if K_req == 1:
            lab = np.zeros(n_p, dtype=int)
        else:
            km = KMeans(n_clusters=K_req, n_init=10, random_state=SEED)
            lab = km.fit_predict(Z)
            lab, merges = enforce_min_size(lab, K_req, min_cluster)
            # Order clusters by mean abundance so the index is interpretable and
            # stable across K: cluster 0 is always the least abundant stratum.
            K = int(lab.max()) + 1
            order = np.argsort(
                [stat.loc[lab == c, "abundance_mean"].mean() for c in range(K)]
            )
            remap = np.empty(K, dtype=int)
            remap[order] = np.arange(K)
            lab = remap[lab]
        # The column is keyed by the REQUESTED K so downstream lookup is stable;
        # the realised count after merging is recorded alongside and may be lower.
        K = int(lab.max()) + 1
        clus[f"k{K_req}"] = lab
        sizes = np.bincount(lab, minlength=K).tolist()
        if merges:
            log(f"    K={K_req}: merged {len(merges)} undersized cluster(s) -> "
                f"{K} realised clusters")
        prof[f"k{K_req}"] = {
            "n_clusters_requested": int(K_req),
            "n_clusters_realised": int(K),
            "n_merges": len(merges),
            "merges": merges,
            "min_cluster_size_enforced": int(min_cluster),
            "sizes": sizes,
            "abundance_mean_by_cluster": [
                float(stat.loc[lab == c, "abundance_mean"].mean()) for c in range(K)
            ],
            "delta_sd_by_cluster": [
                float(stat.loc[lab == c, "delta_sd"].mean()) for c in range(K)
            ],
            "detect_rate_by_cluster": [
                float(stat.loc[lab == c, "detect_rate"].mean()) for c in range(K)
            ],
            "weighted_degree_by_cluster": [
                float(stat.loc[lab == c, "weighted_degree"].mean()) for c in range(K)
            ],
        }
        log(f"  K={K_req:2d} -> {K:2d} realised: sizes {sizes}")
    report["cluster_profiles"] = prof
    report["cluster_feature_views"] = {
        "abundance_statistics": stat_cols,
        "spectral_dims_used": int(min(16, emb.shape[1])),
        "scaling": "each view standardised then divided by sqrt(n_dims) for equal total variance",
    }

    np.savez_compressed(
        DATA / "step5_protein_graph.npz",
        adj_data=adj.data.astype("float32"),
        adj_indices=adj.indices,
        adj_indptr=adj.indptr,
        adj_shape=np.array(adj.shape),
        embedding=emb,
        eigenvalues=evals.astype("float32"),
        proteins=np.array(proteins, dtype=str),
        graph_source=np.array([graph_source]),
    )
    log(f"  wrote {DATA / 'step5_protein_graph.npz'}")
    stat.to_parquet(DATA / "step5_protein_stats.parquet", compression="snappy")
    clus.to_parquet(DATA / "step5_protein_clusters.parquet", compression="snappy")
    log(f"  wrote {DATA / 'step5_protein_clusters.parquet'} {clus.shape}")

    make_graph_figure(emb, clus, stat, graph_source, report)
    S4.write_json(RESULTS / "step5_graph_report.json", report)


def make_graph_figure(emb, clus, stat, graph_source, report) -> None:
    """Embedding scatter, degree distribution and cluster abundance profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4))

    lab = clus["k8"].to_numpy() if "k8" in clus.columns else clus.iloc[:, 1].to_numpy()
    nK = int(lab.max()) + 1
    cmap = plt.get_cmap("tab10")

    ax = axes[0, 0]
    for c in range(nK):
        m = lab == c
        ax.scatter(emb[m, 1], emb[m, 2], s=3, color=cmap(c % 10), lw=0,
                   alpha=0.6, label=f"c{c} (n={int(m.sum())})")
    ax.set_xlabel("spectral dim 2")
    ax.set_ylabel("spectral dim 3")
    ax.set_title(f"Protein graph spectral embedding\n({graph_source}, K=8 clusters)")
    ax.legend(fontsize=5, markerscale=2.2, ncol=2, frameon=False)

    ax = axes[0, 1]
    deg = stat["weighted_degree"].to_numpy()
    pos = deg[deg > 0]
    ax.hist(np.log10(pos), bins=60, color="#3B6EA8", lw=0)
    ax.set_xlabel("log10 weighted degree")
    ax.set_ylabel("proteins")
    ax.set_title(f"Interaction degree\n({int((deg == 0).sum())} isolated of {len(deg)})")

    ax = axes[1, 0]
    ab = [stat.loc[lab == c, "abundance_mean"].to_numpy() for c in range(nK)]
    bp = ax.boxplot(ab, showfliers=False, patch_artist=True, widths=0.65)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(cmap(i % 10))
        b.set_alpha(0.65)
        b.set_linewidth(0.5)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(0.9)
    ax.set_xlabel("protein cluster (ordered by mean abundance)")
    ax.set_ylabel("train mean log2 abundance")
    ax.set_title("Clusters stratify the abundance axis")

    ax = axes[1, 1]
    sd = [stat.loc[lab == c, "delta_sd"].to_numpy() for c in range(nK)]
    bp = ax.boxplot(sd, showfliers=False, patch_artist=True, widths=0.65)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(cmap(i % 10))
        b.set_alpha(0.65)
        b.set_linewidth(0.5)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(0.9)
    ax.set_xlabel("protein cluster")
    ax.set_ylabel("train SD of fold change")
    ax.set_title("...and the response-magnitude axis")

    fig.suptitle(
        "Step 5.2  Protein interaction graph embedding and functional clusters",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"step5_knowledge_graph_embedding.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {FIGURES / 'step5_knowledge_graph_embedding.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["mol3d", "graph", "all"], default="all")
    ap.add_argument("--n-conf", type=int, default=8)
    args = ap.parse_args()

    np.random.seed(SEED)
    FIGURES.mkdir(exist_ok=True)
    log(f"=== Step 5.2: advanced features (part={args.part}) ===")
    if args.part in ("mol3d", "all"):
        build_mol3d(n_conf=args.n_conf)
    if args.part in ("graph", "all"):
        build_graph()
    log("=== Step 5.2 complete ===")


if __name__ == "__main__":
    main()
