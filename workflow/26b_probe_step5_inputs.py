"""Step 5.0 -- probe the inputs Step 5 needs before any expensive work starts.

Answers four questions that determine the shape of the Step-5 scripts:

1. What do the 5,243 protein identifiers actually look like? STRING needs a
   namespace it recognises, so guessing the identifier type would silently
   produce an empty interaction graph.
2. Do the RDKit 3D toolchain (ETKDG embedding, MMFF optimisation, SASA,
   principal moments of inertia, 3D pharmacophore fingerprints) and torch
   import cleanly in this environment?
3. What is the SMILES coverage of the resolved compound table from Step 4?
4. Is the STRING REST API reachable, and what does it return for a handful of
   these identifiers?

Nothing here is modelling; it exists so the later scripts can be written
against the data that is really present rather than against an assumption.
"""

from __future__ import annotations

from repo_paths import REPO_ROOT

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SESSION = REPO_ROOT
WORKFLOW = SESSION / "workflow"
DATA = SESSION / "data"
RESULTS = SESSION / "results"
sys.path.insert(0, str(WORKFLOW))

out: dict = {"step": "5_0_probe"}


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


# ---------------------------------------------------------------------------
section("1. protein identifiers")
# ---------------------------------------------------------------------------
import pyarrow.parquet as pq  # noqa: E402

schema = pq.read_schema(DATA / "log2_train_val.parquet")
cols = [c for c in schema.names if c != "sample_ID"]
print(f"n columns (proteins) = {len(cols)}")
print("first 25:", cols[:25])
print("random 15:", [cols[i] for i in np.random.default_rng(0).integers(0, len(cols), 15)])

pats = {
    "systematic_YxxNNNx (e.g. YAL001C)": sum(
        1 for c in cols if len(c) >= 7 and c[0] == "Y" and c[2].isalpha() and c[3:6].isdigit()
    ),
    "contains ';'": sum(1 for c in cols if ";" in c),
    "contains '_'": sum(1 for c in cols if "_" in c),
    "uniprot-like (6-10 alnum, starts P/Q/O)": sum(
        1 for c in cols if c[:1] in "PQO" and c[1:].isalnum() and 5 <= len(c) <= 10
    ),
    "all upper alnum <= 6 (gene-symbol-like)": sum(
        1 for c in cols if c.isalnum() and c.isupper() and len(c) <= 6
    ),
}
print("\nidentifier pattern counts:")
for k, v in pats.items():
    print(f"  {k:45s} {v}")
out["proteins"] = {
    "n": len(cols),
    "first_25": cols[:25],
    "pattern_counts": pats,
}

# ---------------------------------------------------------------------------
section("2. RDKit 3D toolchain + torch")
# ---------------------------------------------------------------------------
tool: dict = {}
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D, rdFreeSASA
    from rdkit.Chem import rdMolDescriptors

    tool["rdkit_version"] = rdkit.__version__
    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))  # aspirin
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 42
    cid = AllChem.EmbedMolecule(m, ps)
    tool["etkdg_conf_id"] = int(cid)
    ff_ok = AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    tool["mmff_return"] = int(ff_ok)
    tool["npr1"] = float(Descriptors3D.NPR1(m))
    tool["npr2"] = float(Descriptors3D.NPR2(m))
    tool["pmi1"] = float(Descriptors3D.PMI1(m))
    tool["asphericity"] = float(Descriptors3D.Asphericity(m))
    tool["radius_of_gyration"] = float(Descriptors3D.RadiusOfGyration(m))
    radii = rdFreeSASA.classifyAtoms(m)
    tool["freesasa"] = float(rdFreeSASA.CalcSASA(m, radii))
    p3 = rdMolDescriptors.Get3DDistanceMatrix(m)
    tool["dist3d_shape"] = list(p3.shape)
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

    fp2d = Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory)
    tool["gobbi_pharm2d_nbits"] = int(fp2d.GetNumBits())
    tool["gobbi_pharm2d_on"] = int(len(fp2d.GetOnBits()))
    # 3D pharmacophore requires a distance matrix argument
    fp3d = Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory, dMat=p3)
    tool["gobbi_pharm3d_on"] = int(len(fp3d.GetOnBits()))
    tool["ok"] = True
except Exception as exc:  # pragma: no cover - environment probe
    tool["ok"] = False
    tool["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(tool, indent=2))
out["rdkit_3d"] = tool

th: dict = {}
try:
    import torch

    th["torch_version"] = torch.__version__
    th["cuda_available"] = bool(torch.cuda.is_available())
    th["n_gpu"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    th["ok"] = True
except Exception as exc:  # pragma: no cover
    th["ok"] = False
    th["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(th, indent=2))
out["torch"] = th

# ---------------------------------------------------------------------------
section("3. resolved SMILES table from Step 4")
# ---------------------------------------------------------------------------
sm = json.loads((RESULTS / "step4_smiles_resolved.json").read_text(encoding="utf-8"))
print("top-level keys:", list(sm)[:20])
recs = None
for key in ("resolved", "compounds", "records", "rows"):
    if key in sm and isinstance(sm[key], list):
        recs = sm[key]
        break
if recs is None:
    for k, v in sm.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            recs = v
            print(f"  (using list under key '{k}')")
            break
print(f"n records = {len(recs) if recs else 0}")
if recs:
    print("record keys:", sorted(recs[0]))
    smk = [k for k in recs[0] if "smiles" in k.lower()]
    print("smiles-like keys:", smk)
    n_smiles = sum(1 for r in recs if any(r.get(k) for k in smk))
    print(f"records with a non-empty SMILES: {n_smiles}/{len(recs)}")
    for r in recs[:5]:
        print("  ", {k: (str(v)[:60] if v is not None else None) for k, v in r.items()})
    out["smiles"] = {
        "n_records": len(recs),
        "keys": sorted(recs[0]),
        "smiles_keys": smk,
        "n_with_smiles": n_smiles,
    }

mf = pd.read_parquet(DATA / "step4_mol_features.parquet")
print(f"\nstep4_mol_features: {mf.shape}")
print("  non-fp columns:", [c for c in mf.columns if not c.startswith("fp_")][:60])
out["mol_features"] = {
    "shape": list(mf.shape),
    "n_fp_bits": int(sum(c.startswith("fp_") for c in mf.columns)),
    "non_fp_columns": [c for c in mf.columns if not c.startswith("fp_")],
}

# ---------------------------------------------------------------------------
section("4. STRING REST API reachability")
# ---------------------------------------------------------------------------
st: dict = {}
try:
    import requests

    r = requests.get("https://string-db.org/api/tsv/version", timeout=30)
    st["version_endpoint_status"] = r.status_code
    st["version_body"] = r.text.strip()[:200]

    probe_ids = cols[:5]
    r2 = requests.post(
        "https://string-db.org/api/tsv/get_string_ids",
        data={
            "identifiers": "\r".join(probe_ids),
            "species": 4932,  # Saccharomyces cerevisiae
            "limit": 1,
            "caller_identity": "goai-virtual-cell_step5",
        },
        timeout=60,
    )
    st["map_status"] = r2.status_code
    st["map_body_head"] = r2.text.strip().splitlines()[:8]
    st["probe_ids"] = probe_ids
    st["ok"] = r.status_code == 200 and r2.status_code == 200
except Exception as exc:  # pragma: no cover
    st["ok"] = False
    st["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(st, indent=2))
out["string_api"] = st

RESULTS.mkdir(exist_ok=True)
(RESULTS / "step5_probe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\nwrote {RESULTS / 'step5_probe.json'}")
