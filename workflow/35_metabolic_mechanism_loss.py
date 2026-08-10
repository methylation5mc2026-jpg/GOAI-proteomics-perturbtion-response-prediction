#!/usr/bin/env python
"""Step 6.2 -- iMM904 metabolic flux balance and protein complex stoichiometry loss.

Provides ``MechanismLoss``, a PyTorch loss module combining four terms:

  a) masked MSE on the fold-change Delta
  b) masked Pearson correlation loss (per sample)
  c) protein-complex co-response penalty
        sum_{(i,j) in E_complex} w_ij * (dhat_i - dhat_j)^2
  d) metabolic flux conservation penalty
        || S @ v(Yhat) ||_2^2

Biological grounding
--------------------
*Complex co-response.* Subunits of an obligate protein complex are co-regulated
and are degraded when unassembled, so their abundances move together under
perturbation. This is a well-established proteomic regularity, and it gives a
smoothness prior over the response of proteins that share a complex. Complex
memberships are taken from a curated set of yeast complexes (20S/19S
proteasome, SAGA, RNA Pol I/II/III, cytosolic ribosome, TRiC/CCT chaperonin,
V-ATPase, COPI/COPII, eIF3, TFIID, exosome, ...) restricted to subunits that
are actually measured in this proteome.

*Flux conservation.* In iMM904-style stoichiometric models a metabolic steady
state satisfies S @ v = 0, where S is the metabolite-by-reaction stoichiometry
matrix and v the flux vector. We do not observe fluxes; we observe enzyme
abundance. We therefore use the standard enzyme-capacity surrogate: the
predicted relative flux through a reaction is taken to be proportional to the
predicted relative abundance of its catalysing enzyme(s). The penalty
||S v(Yhat)||^2 then asks that the *predicted enzyme response pattern* remain
compatible with metabolite mass balance -- i.e. that the model not predict a
large increase in the enzyme producing a metabolite with a simultaneous large
decrease in every enzyme consuming it.

This surrogate is an approximation and is documented as such: enzyme abundance
bounds capacity rather than determining flux, and post-translational regulation
is not captured. The penalty is used as a weak regulariser (small coefficient),
not as a hard constraint, precisely because of that.

Outputs
-------
data/step6_mechanism_structures.npz    complex edges + stoichiometry matrix
results/step6_mechanism_loss_report.json
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SESSION = REPO_ROOT
DATA = SESSION / "data"
RESULTS = SESSION / "results"
SEED = 42


# ---------------------------------------------------------------------------
# Curated yeast protein complexes (standard gene names).
# Sources: SGD / CYC2008-style curated complex memberships. Only subunits that
# are measured in this proteome are retained at build time.
# ---------------------------------------------------------------------------
YEAST_COMPLEXES: dict[str, list[str]] = {
    "proteasome_20S_core": [
        "PRE1", "PRE2", "PRE3", "PRE4", "PRE5", "PRE6", "PRE7", "PRE8", "PRE9",
        "PUP1", "PUP2", "PUP3", "SCL1",
    ],
    "proteasome_19S_regulatory": [
        "RPT1", "RPT2", "RPT3", "RPT4", "RPT5", "RPT6", "RPN1", "RPN2", "RPN3",
        "RPN5", "RPN6", "RPN7", "RPN8", "RPN9", "RPN10", "RPN11", "RPN12",
    ],
    "SAGA": [
        "SPT3", "SPT7", "SPT8", "SPT20", "ADA2", "ADA3", "GCN5", "TAF5", "TAF6",
        "TAF9", "TAF10", "TAF12", "TRA1", "SGF29", "SGF73", "UBP8", "SUS1", "NGG1",
    ],
    "RNA_Pol_II": [
        "RPB1", "RPB2", "RPB3", "RPB4", "RPB5", "RPB7", "RPB8", "RPB9", "RPB10",
        "RPB11", "RPO21", "RPC10",
    ],
    "RNA_Pol_I": ["RPA190", "RPA135", "RPA49", "RPA43", "RPA34", "RPA14", "RPA12"],
    "RNA_Pol_III": ["RET1", "RPC40", "RPC53", "RPC82", "RPC34", "RPC31", "RPC19", "RPC11"],
    "TRiC_CCT_chaperonin": ["CCT2", "CCT3", "CCT4", "CCT5", "CCT6", "CCT7", "CCT8", "TCP1"],
    "V_ATPase": [
        "VMA1", "VMA2", "VMA4", "VMA5", "VMA6", "VMA7", "VMA8", "VMA10", "VMA13",
        "VPH1", "TFP1", "TFP3",
    ],
    "F1F0_ATP_synthase": [
        "ATP1", "ATP2", "ATP3", "ATP4", "ATP5", "ATP7", "ATP14", "ATP15", "ATP16",
        "ATP17", "ATP18", "ATP20",
    ],
    "eIF3": ["PRT1", "NIP1", "RPG1", "TIF34", "TIF35", "TIF32", "HCR1"],
    "TFIID": ["TAF1", "TAF2", "TAF4", "TAF5", "TAF6", "TAF7", "TAF8", "TAF11",
              "TAF12", "TAF13", "TAF14", "SPT15"],
    "exosome": ["RRP4", "RRP6", "RRP40", "RRP41", "RRP42", "RRP43", "RRP45",
                "RRP46", "MTR3", "CSL4", "DIS3"],
    "COPI": ["COP1", "SEC26", "SEC27", "SEC21", "RET2", "RET3", "SEC28"],
    "COPII": ["SEC13", "SEC31", "SEC23", "SEC24", "SEC16", "SAR1"],
    "chaperonin_HSP90_system": ["HSP82", "HSC82", "STI1", "SBA1", "CDC37", "CPR6", "CPR7"],
    "ribosome_40S": [f"RPS{i}{s}" for i in range(1, 32) for s in ("A", "B")],
    "ribosome_60S": [f"RPL{i}{s}" for i in range(1, 44) for s in ("A", "B")],
    "MCM_helicase": ["MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7"],
    "cytoskeleton_tubulin": ["TUB1", "TUB2", "TUB3", "TUB4"],
    "CCR4_NOT": ["CCR4", "NOT1", "NOT2", "NOT3", "NOT4", "NOT5", "CAF40", "CAF130", "POP2"],
}


# ---------------------------------------------------------------------------
# iMM904-style metabolic reactions: (reaction id, {metabolite: coefficient},
# [catalysing enzymes]). Coefficients follow the convention that products are
# positive and substrates negative. This is a curated core-metabolism subset of
# iMM904 covering glycolysis, fermentation, the TCA cycle, the pentose
# phosphate pathway and ergosterol biosynthesis -- the pathways most directly
# engaged by the chemical panel (antifungal azoles hit ERG11, uncouplers hit
# oxidative phosphorylation, and so on).
# ---------------------------------------------------------------------------
IMM904_REACTIONS: list[tuple[str, dict[str, float], list[str]]] = [
    # --- glycolysis ---
    ("HEX1",  {"glc_D": -1.0, "g6p": 1.0},                 ["HXK1", "HXK2", "GLK1"]),
    ("PGI",   {"g6p": -1.0, "f6p": 1.0},                   ["PGI1"]),
    ("PFK",   {"f6p": -1.0, "fdp": 1.0},                   ["PFK1", "PFK2"]),
    ("FBA",   {"fdp": -1.0, "dhap": 1.0, "g3p": 1.0},      ["FBA1"]),
    ("TPI",   {"dhap": -1.0, "g3p": 1.0},                  ["TPI1"]),
    ("GAPD",  {"g3p": -1.0, "x13dpg": 1.0},                ["TDH1", "TDH2", "TDH3"]),
    ("PGK",   {"x13dpg": -1.0, "x3pg": 1.0},               ["PGK1"]),
    ("PGM",   {"x3pg": -1.0, "x2pg": 1.0},                 ["GPM1"]),
    ("ENO",   {"x2pg": -1.0, "pep": 1.0},                  ["ENO1", "ENO2", "ERR1", "ERR2"]),
    ("PYK",   {"pep": -1.0, "pyr": 1.0},                   ["CDC19", "PYK2"]),
    # --- fermentation ---
    ("PDC",   {"pyr": -1.0, "acald": 1.0},                 ["PDC1", "PDC5", "PDC6"]),
    ("ALCD",  {"acald": -1.0, "etoh": 1.0},                ["ADH1", "ADH2", "ADH3", "ADH4", "ADH5"]),
    ("ALDD",  {"acald": -1.0, "ac": 1.0},                  ["ALD2", "ALD3", "ALD4", "ALD5", "ALD6"]),
    ("ACS",   {"ac": -1.0, "accoa": 1.0},                  ["ACS1", "ACS2"]),
    # --- TCA cycle ---
    ("PDH",   {"pyr": -1.0, "accoa": 1.0},                 ["PDA1", "PDB1", "LAT1", "LPD1"]),
    ("CS",    {"accoa": -1.0, "oaa": -1.0, "cit": 1.0},    ["CIT1", "CIT2", "CIT3"]),
    ("ACONT", {"cit": -1.0, "icit": 1.0},                  ["ACO1", "ACO2"]),
    ("ICDH",  {"icit": -1.0, "akg": 1.0},                  ["IDH1", "IDH2", "IDP1", "IDP2", "IDP3"]),
    ("AKGD",  {"akg": -1.0, "succoa": 1.0},                ["KGD1", "KGD2", "LPD1"]),
    ("SUCOAS", {"succoa": -1.0, "succ": 1.0},              ["LSC1", "LSC2"]),
    ("SUCD",  {"succ": -1.0, "fum": 1.0},                  ["SDH1", "SDH2", "SDH3", "SDH4"]),
    ("FUM",   {"fum": -1.0, "mal_L": 1.0},                 ["FUM1"]),
    ("MDH",   {"mal_L": -1.0, "oaa": 1.0},                 ["MDH1", "MDH2", "MDH3"]),
    # --- pentose phosphate ---
    ("G6PDH", {"g6p": -1.0, "x6pgl": 1.0},                 ["ZWF1"]),
    ("PGL",   {"x6pgl": -1.0, "x6pgc": 1.0},               ["SOL3", "SOL4"]),
    ("GND",   {"x6pgc": -1.0, "ru5p_D": 1.0},              ["GND1", "GND2"]),
    ("RPI",   {"ru5p_D": -1.0, "r5p": 1.0},                ["RKI1"]),
    ("RPE",   {"ru5p_D": -1.0, "xu5p_D": 1.0},             ["RPE1"]),
    ("TKT1",  {"r5p": -1.0, "xu5p_D": -1.0, "s7p": 1.0, "g3p": 1.0}, ["TKL1", "TKL2"]),
    ("TALA",  {"s7p": -1.0, "g3p": -1.0, "e4p": 1.0, "f6p": 1.0},    ["TAL1", "NQM1"]),
    # --- ergosterol biosynthesis (azole / polyene target pathway) ---
    ("HMGCOAR", {"hmgcoa": -1.0, "mev_R": 1.0},            ["HMG1", "HMG2"]),
    ("MEVK",  {"mev_R": -1.0, "x5pmev": 1.0},              ["ERG12"]),
    ("PMEVK", {"x5pmev": -1.0, "x5dpmev": 1.0},            ["ERG8"]),
    ("DPMVD", {"x5dpmev": -1.0, "ipdp": 1.0},              ["MVD1"]),
    ("IPDDI", {"ipdp": -1.0, "dmpp": 1.0},                 ["IDI1"]),
    ("FRTT",  {"ipdp": -1.0, "dmpp": -1.0, "frdp": 1.0},   ["ERG20"]),
    ("SQLS",  {"frdp": -2.0, "sql": 1.0},                  ["ERG9"]),
    ("SQLE",  {"sql": -1.0, "sql23epx": 1.0},              ["ERG1"]),
    ("LNS",   {"sql23epx": -1.0, "lanost": 1.0},           ["ERG7"]),
    ("C14DM", {"lanost": -1.0, "x44mzym": 1.0},            ["ERG11"]),
    ("C14STR", {"x44mzym": -1.0, "zymst": 1.0},            ["ERG24", "ERG25", "ERG26", "ERG27"]),
    ("C24STR", {"zymst": -1.0, "fecost": 1.0},             ["ERG6"]),
    ("C8ISO", {"fecost": -1.0, "epist": 1.0},              ["ERG2", "ERG3"]),
    ("ERGSTt", {"epist": -1.0, "ergst": 1.0},              ["ERG4", "ERG5"]),
]


# ---------------------------------------------------------------------------
# Structure construction
# ---------------------------------------------------------------------------
def build_structures(proteins: list[str]) -> dict:
    """Build complex edges and the metabolite x reaction stoichiometry matrix.

    Parameters
    ----------
    proteins
        Ordered protein column names of the response matrix.

    Returns
    -------
    dict with
        ``edge_i``, ``edge_j``, ``edge_w`` : complex co-response edges
        ``S``                : (n_metabolites, n_reactions) float32
        ``rxn_enzyme_idx``   : (n_reactions, max_enz) int64, -1 padded
        ``rxn_enzyme_mask``  : (n_reactions, max_enz) float32
    """
    pidx = {p.upper(): i for i, p in enumerate(proteins)}

    # ---- complex co-response edges ----
    edge_i, edge_j, edge_w, comp_report = [], [], [], {}
    for cname, members in YEAST_COMPLEXES.items():
        present = sorted({pidx[m.upper()] for m in members if m.upper() in pidx})
        comp_report[cname] = {"n_curated": len(set(members)), "n_measured": len(present)}
        if len(present) < 2:
            continue
        # Weight each edge by 1/(k-1) so a large complex does not dominate the
        # penalty purely through its quadratic edge count.
        w = 1.0 / (len(present) - 1)
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                edge_i.append(present[a])
                edge_j.append(present[b])
                edge_w.append(w)

    # ---- stoichiometry matrix ----
    mets, rxns_kept = [], []
    met_index: dict[str, int] = {}
    rxn_enz: list[list[int]] = []
    rxn_report = {}
    for rid, stoich, enzymes in IMM904_REACTIONS:
        eidx = sorted({pidx[e.upper()] for e in enzymes if e.upper() in pidx})
        rxn_report[rid] = {"n_curated_enzymes": len(set(enzymes)), "n_measured": len(eidx)}
        if not eidx:  # no measured catalyst -> reaction carries no observable signal
            continue
        rxns_kept.append((rid, stoich))
        rxn_enz.append(eidx)
        for m in stoich:
            if m not in met_index:
                met_index[m] = len(mets)
                mets.append(m)

    n_met, n_rxn = len(mets), len(rxns_kept)
    S = np.zeros((n_met, n_rxn), dtype=np.float32)
    for j, (_rid, stoich) in enumerate(rxns_kept):
        for m, c in stoich.items():
            S[met_index[m], j] = c

    # Drop metabolites that appear in only one retained reaction: mass balance
    # over such a metabolite is uninformative (a dead-end that would simply
    # penalise any nonzero flux through the single reaction touching it).
    deg = (S != 0).sum(axis=1)
    keep = deg >= 2
    S = S[keep]
    mets = [m for m, k in zip(mets, keep) if k]

    max_e = max(len(e) for e in rxn_enz) if rxn_enz else 1
    enz_idx = np.full((n_rxn, max_e), -1, dtype=np.int64)
    enz_msk = np.zeros((n_rxn, max_e), dtype=np.float32)
    for j, e in enumerate(rxn_enz):
        enz_idx[j, :len(e)] = e
        enz_msk[j, :len(e)] = 1.0

    return {
        "edge_i": np.asarray(edge_i, dtype=np.int64),
        "edge_j": np.asarray(edge_j, dtype=np.int64),
        "edge_w": np.asarray(edge_w, dtype=np.float32),
        "S": S,
        "metabolites": mets,
        "reactions": [r for r, _ in rxns_kept],
        "rxn_enzyme_idx": enz_idx,
        "rxn_enzyme_mask": enz_msk,
        "complex_report": comp_report,
        "reaction_report": rxn_report,
    }


# ---------------------------------------------------------------------------
# The loss module
# ---------------------------------------------------------------------------
class MechanismLoss(nn.Module):
    """Mechanism-guided composite loss on predicted fold-change Delta.

    Parameters
    ----------
    struct
        Output of :func:`build_structures`.
    w_mse, w_pcc, w_complex, w_flux
        Term coefficients. The two mechanism terms default to small values:
        they are regularisers expressing a prior, not data-fitting objectives.
    """

    def __init__(self, struct: dict, w_mse: float = 1.0, w_pcc: float = 0.5,
                 w_complex: float = 0.02, w_flux: float = 0.01,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.w_mse = float(w_mse)
        self.w_pcc = float(w_pcc)
        self.w_complex = float(w_complex)
        self.w_flux = float(w_flux)
        dev = torch.device(device)
        self.register_buffer("edge_i", torch.as_tensor(struct["edge_i"], device=dev))
        self.register_buffer("edge_j", torch.as_tensor(struct["edge_j"], device=dev))
        self.register_buffer("edge_w", torch.as_tensor(struct["edge_w"], device=dev))
        self.register_buffer("S", torch.as_tensor(struct["S"], device=dev))
        self.register_buffer("enz_idx",
                             torch.as_tensor(np.clip(struct["rxn_enzyme_idx"], 0, None),
                                             device=dev))
        self.register_buffer("enz_msk", torch.as_tensor(struct["rxn_enzyme_mask"], device=dev))

    # -- individual terms ---------------------------------------------------
    @staticmethod
    def masked_mse(pred: torch.Tensor, true: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """Mean squared error over finite cells only."""
        n = mask.sum()
        if n <= 0:
            return pred.sum() * 0.0
        diff = (pred - true) * mask
        return (diff * diff).sum() / n

    @staticmethod
    def masked_pcc_loss(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor,
                        min_n: float = 3.0, eps: float = 1e-8) -> torch.Tensor:
        """1 - mean per-sample masked Pearson correlation.

        Samples with fewer than ``min_n`` finite cells or zero variance are
        excluded, mirroring the harness convention (which returns NaN rather
        than 0 for such slices) instead of silently scoring them as anti-correlated.
        """
        n = mask.sum(dim=1)
        ok = n >= min_n
        if not bool(ok.any()):
            return pred.sum() * 0.0
        nz = n.clamp(min=1.0)
        mp = (pred * mask).sum(1) / nz
        mt = (true * mask).sum(1) / nz
        dp = (pred - mp[:, None]) * mask
        dt = (true - mt[:, None]) * mask
        num = (dp * dt).sum(1)
        den = torch.sqrt((dp * dp).sum(1) * (dt * dt).sum(1) + eps)
        r = num / den.clamp(min=eps)
        var_ok = ((dp * dp).sum(1) > eps) & ((dt * dt).sum(1) > eps)
        sel = ok & var_ok
        if not bool(sel.any()):
            return pred.sum() * 0.0
        return 1.0 - r[sel].mean()

    def complex_penalty(self, pred: torch.Tensor) -> torch.Tensor:
        """Co-response penalty over protein-complex subunit pairs."""
        if self.edge_i.numel() == 0:
            return pred.sum() * 0.0
        d = pred[:, self.edge_i] - pred[:, self.edge_j]
        return ((d * d) * self.edge_w[None, :]).sum(1).mean()

    def flux_penalty(self, pred: torch.Tensor) -> torch.Tensor:
        """|| S @ v(Yhat) ||^2 with v the enzyme-capacity flux surrogate.

        The relative flux of reaction j is taken as the mean predicted
        log2 fold-change of its measured catalysing enzymes, which is the
        standard enzyme-capacity proxy: a reaction cannot carry more flux than
        its catalyst supports. Mass balance then requires the resulting flux
        pattern to be compatible with S v = 0.
        """
        if self.S.numel() == 0:
            return pred.sum() * 0.0
        # (batch, n_rxn, max_enz) -> mean over measured enzymes
        gath = pred[:, self.enz_idx]                       # (B, R, E)
        msk = self.enz_msk[None, :, :]
        v = (gath * msk).sum(-1) / msk.sum(-1).clamp(min=1.0)   # (B, R)
        imb = v @ self.S.T                                  # (B, n_met)
        return (imb * imb).sum(1).mean()

    # -- composite ----------------------------------------------------------
    def forward(self, pred: torch.Tensor, true: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """Return ``(total_loss, per_term_dict)``.

        ``mask`` is 1.0 on cells where ``true`` is measured. If omitted it is
        derived from finiteness of ``true``. NaNs in ``true`` are zeroed after
        masking so they cannot propagate into the gradient.
        """
        if mask is None:
            mask = torch.isfinite(true).float()
        true = torch.nan_to_num(true, nan=0.0, posinf=0.0, neginf=0.0)
        t_mse = self.masked_mse(pred, true, mask)
        t_pcc = self.masked_pcc_loss(pred, true, mask)
        t_cpx = self.complex_penalty(pred)
        t_flx = self.flux_penalty(pred)
        total = (self.w_mse * t_mse + self.w_pcc * t_pcc
                 + self.w_complex * t_cpx + self.w_flux * t_flx)
        return total, {"mse": float(t_mse.detach()), "pcc_loss": float(t_pcc.detach()),
                       "complex": float(t_cpx.detach()), "flux": float(t_flx.detach()),
                       "total": float(total.detach())}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _numpy_reference(pred, true, mask, struct) -> dict:
    """Independent NumPy implementation of every term, for cross-checking."""
    n = mask.sum()
    mse = float((((pred - true) * mask) ** 2).sum() / n)

    rs = []
    for i in range(pred.shape[0]):
        m = mask[i] > 0
        if m.sum() < 3:
            continue
        a, b = pred[i][m], true[i][m]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    pcc = 1.0 - float(np.mean(rs)) if rs else 0.0

    d = pred[:, struct["edge_i"]] - pred[:, struct["edge_j"]]
    cpx = float((((d ** 2) * struct["edge_w"][None, :]).sum(1)).mean())

    ei, em = struct["rxn_enzyme_idx"], struct["rxn_enzyme_mask"]
    g = pred[:, np.clip(ei, 0, None)]
    v = (g * em[None]).sum(-1) / np.clip(em.sum(-1), 1.0, None)
    imb = v @ struct["S"].T
    flx = float((imb ** 2).sum(1).mean())
    return {"mse": mse, "pcc_loss": pcc, "complex": cpx, "flux": flx}


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("=" * 78)
    print("Step 6.2  iMM904 flux + protein complex stoichiometry loss")
    print("=" * 78, flush=True)

    prot = pd.read_parquet(DATA / "step5_protein_stats.parquet")
    proteins = [str(p) for p in prot["protein"].astype(str)]
    print(f"[1] proteome: {len(proteins)} measured proteins", flush=True)

    st = build_structures(proteins)
    n_edges = len(st["edge_i"])
    print(f"[2] complex co-response edges: {n_edges} over "
          f"{sum(1 for v in st['complex_report'].values() if v['n_measured'] >= 2)} "
          f"complexes with >=2 measured subunits", flush=True)
    for c, r in sorted(st["complex_report"].items()):
        print(f"      {c:<28} {r['n_measured']:>3}/{r['n_curated']:<3} measured", flush=True)
    print(f"[3] stoichiometry matrix S: {st['S'].shape} "
          f"({len(st['metabolites'])} balanced metabolites x "
          f"{len(st['reactions'])} reactions with >=1 measured enzyme)", flush=True)
    dropped = [r for r, v in st["reaction_report"].items() if v["n_measured"] == 0]
    if dropped:
        print(f"      reactions dropped for having no measured enzyme: {dropped}", flush=True)

    np.savez_compressed(
        DATA / "step6_mechanism_structures.npz",
        edge_i=st["edge_i"], edge_j=st["edge_j"], edge_w=st["edge_w"], S=st["S"],
        rxn_enzyme_idx=st["rxn_enzyme_idx"], rxn_enzyme_mask=st["rxn_enzyme_mask"],
        metabolites=np.array(st["metabolites"], dtype=str),
        reactions=np.array(st["reactions"], dtype=str))
    print("    saved data/step6_mechanism_structures.npz", flush=True)

    # ---------------- term-by-term verification ----------------
    print("[4] verifying every term against an independent NumPy reference ...", flush=True)
    rng = np.random.default_rng(SEED)
    B, P = 16, len(proteins)
    pred_np = rng.normal(0, 0.5, (B, P)).astype(np.float32)
    true_np = (pred_np * 0.6 + rng.normal(0, 0.4, (B, P))).astype(np.float32)
    mask_np = (rng.random((B, P)) > 0.27).astype(np.float32)  # ~27% missing, as observed
    ref = _numpy_reference(pred_np.astype(np.float64), true_np.astype(np.float64),
                           mask_np.astype(np.float64), st)

    loss = MechanismLoss(st, device="cpu")
    pt = torch.tensor(pred_np, requires_grad=True)
    tt = torch.tensor(true_np)
    mt = torch.tensor(mask_np)
    _total, terms = loss(pt, tt, mt)
    devs = {k: abs(terms[k] - ref[k]) for k in ref}
    for k in ref:
        print(f"      {k:<10} torch={terms[k]:>14.8f}  numpy={ref[k]:>14.8f}  "
              f"|dev|={devs[k]:.3e}", flush=True)
    max_dev = max(devs.values())
    tol = 1e-3
    assert max_dev < tol, f"MechanismLoss disagrees with the NumPy reference: {devs}"
    print(f"    max deviation {max_dev:.3e} < {tol} -- all four terms verified", flush=True)

    # ---------------- gradient sanity ----------------
    print("[5] gradient check ...", flush=True)
    _total.backward()
    g = pt.grad.detach().numpy()
    assert np.isfinite(g).all(), "non-finite gradient"
    print(f"    gradient finite; |g| mean={np.abs(g).mean():.3e} max={np.abs(g).max():.3e}",
          flush=True)

    # ---------------- convergence check ----------------
    print("[6] convergence check: fitting a free Delta tensor ...", flush=True)
    z = torch.zeros(B, P, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.05)
    traj = []
    for ep in range(300):
        opt.zero_grad()
        tot, tm = loss(z, tt, mt)
        tot.backward()
        opt.step()
        if ep % 30 == 0 or ep == 299:
            traj.append({"epoch": ep, **tm})
            print(f"      epoch {ep:>3}  total={tm['total']:.5f}  mse={tm['mse']:.5f}  "
                  f"pcc_loss={tm['pcc_loss']:.5f}  complex={tm['complex']:.5f}  "
                  f"flux={tm['flux']:.5f}", flush=True)
    assert traj[-1]["total"] < traj[0]["total"], "MechanismLoss did not converge"
    print(f"    total loss {traj[0]['total']:.5f} -> {traj[-1]['total']:.5f} "
          f"({100 * (1 - traj[-1]['total'] / traj[0]['total']):.1f}% reduction)", flush=True)

    # ---------------- ablation: does each term move independently? ----------
    # NOTE: the starting point must be random, not all-zero. At Delta = 0 the
    # Pearson, complex and flux terms are all degenerately zero (no variance, no
    # subunit difference, no flux), so a zero init would make three of these four
    # checks vacuously pass. We start from noise so each term begins strictly
    # positive and a genuine decrease has to be demonstrated.
    print("[7] term-isolation check (each coefficient alone, random init) ...", flush=True)
    iso = {}
    init = torch.tensor(rng.normal(0, 0.5, (B, P)).astype(np.float32))
    for name, kw in (("mse_only", dict(w_mse=1, w_pcc=0, w_complex=0, w_flux=0)),
                     ("pcc_only", dict(w_mse=0, w_pcc=1, w_complex=0, w_flux=0)),
                     ("complex_only", dict(w_mse=0, w_pcc=0, w_complex=1, w_flux=0)),
                     ("flux_only", dict(w_mse=0, w_pcc=0, w_complex=0, w_flux=1))):
        L = MechanismLoss(st, **kw)
        zz = init.clone().requires_grad_(True)
        o = torch.optim.Adam([zz], lr=0.05)
        first = last = None
        for ep in range(120):
            o.zero_grad()
            tt2, tm2 = L(zz, tt, mt)
            tt2.backward()
            o.step()
            if ep == 0:
                first = tm2["total"]
            last = tm2["total"]
        iso[name] = {"start": first, "end": last,
                     "started_strictly_positive": bool(first > 1e-9),
                     "decreased": bool(last < first - 1e-9)}
        print(f"      {name:<13} {first:.6f} -> {last:.6f}  "
              f"{'OK' if iso[name]['decreased'] else 'NOT DECREASING'}", flush=True)
    bad = {k: v for k, v in iso.items()
           if not (v["decreased"] and v["started_strictly_positive"])}
    assert not bad, f"a term was vacuous or failed to optimise: {bad}"

    report = {
        "step": "6_2_metabolic_mechanism_loss",
        "seed": SEED,
        "n_proteins": len(proteins),
        "n_complexes_curated": len(YEAST_COMPLEXES),
        "n_complexes_with_2plus_measured_subunits":
            int(sum(1 for v in st["complex_report"].values() if v["n_measured"] >= 2)),
        "n_complex_edges": int(n_edges),
        "complex_membership": st["complex_report"],
        "edge_weighting": ("each edge weighted 1/(k-1) for a complex with k measured "
                           "subunits, so a large complex does not dominate the penalty "
                           "through its quadratic edge count"),
        "n_reactions_curated": len(IMM904_REACTIONS),
        "n_reactions_with_measured_enzyme": len(st["reactions"]),
        "reactions_dropped_no_measured_enzyme": dropped,
        "n_balanced_metabolites": len(st["metabolites"]),
        "stoichiometry_shape": list(st["S"].shape),
        "reaction_coverage": st["reaction_report"],
        "flux_surrogate_note": (
            "Fluxes are not observed. v_j is taken as the mean predicted log2 "
            "fold-change of reaction j's measured catalysing enzymes -- the standard "
            "enzyme-capacity proxy. The penalty ||S v||^2 therefore asks that the "
            "predicted enzyme response pattern stay compatible with metabolite mass "
            "balance. This is an approximation: enzyme abundance bounds capacity "
            "rather than determining flux, and post-translational regulation is not "
            "captured. It is consequently applied as a weak regulariser (default "
            "coefficient 0.01), never as a hard constraint."),
        "dead_end_metabolite_policy": (
            "metabolites touching fewer than 2 retained reactions are dropped, because "
            "mass balance over a dead-end would merely penalise any nonzero flux"),
        "verification": {
            "torch_vs_numpy_max_abs_deviation": float(max_dev),
            "tolerance": tol,
            "per_term_deviation": {k: float(v) for k, v in devs.items()},
            "gradient_finite": True,
            "gradient_abs_mean": float(np.abs(g).mean()),
        },
        "convergence_trajectory": traj,
        "term_isolation": iso,
        "default_coefficients": {"w_mse": 1.0, "w_pcc": 0.5,
                                 "w_complex": 0.02, "w_flux": 0.01},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (RESULTS / "step6_mechanism_loss_report.json").write_text(json.dumps(report, indent=2))
    print("[8] -> results/step6_mechanism_loss_report.json", flush=True)
    print(f"DONE in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
