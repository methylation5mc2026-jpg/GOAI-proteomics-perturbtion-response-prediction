#!/usr/bin/env python
"""Step 6.3b -- how much of the cross-attention modulation is actually used?

The Step-6.3 figure shows that chemistry does move the attention placed on
STRING edges. That is necessary but not sufficient: a model could satisfy it
with a single global departure from static attention that every compound shares,
which would be a constant, not a chemistry-dependent, effect.

This script loads the trained fold-0 checkpoint and measures the *rank* of the
modulation directly from the mode-mixture weights alpha(M) = softmax(MLP(M)):

* per-compound mixture entropy -- how many of the R modes a single compound uses;
* between-compound spread of alpha -- whether different compounds select
  different modes at all;
* participation ratio of the alpha matrix's singular values -- the effective
  number of independent chemical directions the attention actually responds to.

A participation ratio near 1 would mean the "dynamic" attention had collapsed to
one shared pattern. Reporting it either way keeps the architectural claim honest.

Outputs
-------
results/step6_attention_rank_diagnostics.json
figures/step6_cross_attention_weights.png   (regenerated, richer)
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SESSION = REPO_ROOT
WORKFLOW = SESSION / "workflow"
DATA = SESSION / "data"
RESULTS = SESSION / "results"
FIGURES = SESSION / "figures"
MODELS5 = WORKFLOW / "models_step5"
SEED = 42
sys.path.insert(0, str(WORKFLOW))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    xa = load_module(WORKFLOW / "36_dynamic_cross_attention_gnn.py", "xa36")
    # Script 36 registers these inside its own main(), which we do not call, so importing
    # it alone leaves sys.modules without them. Load them here rather than relying on a
    # side effect that only happens during a training run.
    dl = sys.modules.get("dl24") or load_module(
        WORKFLOW / "24_train_deep_learning.py", "dl24")
    st30 = sys.modules.get("st30") or load_module(
        WORKFLOW / "30_gnn_and_cluster_stacking.py", "st30")
    S4 = sys.modules["step4_common"]

    ckpt_p = MODELS5 / "xattn_fold0.pt"
    if not ckpt_p.exists():
        raise SystemExit(f"{ckpt_p} absent; run 36_dynamic_cross_attention_gnn.py first")

    G = st30.load_graph()
    nbr_idx, nbr_msk = st30.topk_neighbours(G["adj"], k=12)
    ctx = S4.load_context()
    meta, Y, C = ctx["meta"], ctx["Y"], ctx["C"]
    proteins = ctx["proteins"]
    lc = load_module(WORKFLOW / "29_lcgo_oof_matrix.py", "lcgo29b")
    train_mask = ctx["masks"][ctx["VS"].TRAIN_SPLIT]
    folds = lc.build_folds(meta, train_mask, seed=SEED)
    fit_mask = folds["fits"][0]

    chem_all = meta[S4.CHEM_COL].astype(str).to_numpy()
    chem_names = sorted(set(chem_all))
    fit_chems = set(meta.loc[fit_mask, S4.CHEM_COL].astype(str))
    Mf, mcols = xa.build_chem_context(chem_names, fit_chems)

    fz = dl.SampleFeaturizer(meta, C, Y, fit_mask)
    net = xa.XAttnNet(fz, G["embedding"], nbr_idx, nbr_msk, Mf.shape[1],
                      width=384, n_blocks=3, d_prot=64, n_heads=4, n_modes=8)
    checkpoint = ckpt_p.resolve()
    if not checkpoint.is_relative_to(MODELS5.resolve()):
        raise ValueError("Model checkpoint escaped the expected model directory")
    state = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    net.load_state_dict(state)
    net.eval()

    # ---- mode-mixture weights alpha(M) ----
    with torch.no_grad():
        raw = net.xattn.W_M(torch.from_numpy(Mf))            # (U, R*h)
        alpha = torch.softmax(raw.reshape(len(chem_names), net.xattn.R, net.xattn.h),
                              dim=1).numpy()                  # (U, R, h)
    A = alpha.mean(axis=2)                                    # (U, R) head-averaged
    R = A.shape[1]

    ent = -(A * np.log(np.clip(A, 1e-12, None))).sum(1)
    ent_uniform = float(np.log(R))
    spread = float(A.std(axis=0).mean())

    Ac = A - A.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Ac, compute_uv=False)
    p = (sv ** 2) / max(float((sv ** 2).sum()), 1e-12)
    participation = float(1.0 / np.clip((p ** 2).sum(), 1e-12, None))

    print("=" * 74)
    print("Cross-attention modulation rank diagnostics (fold 0)")
    print("=" * 74)
    print(f"  modes R                                : {R}")
    print(f"  mean per-compound mixture entropy      : {ent.mean():.4f} "
          f"(uniform over {R} modes = {ent_uniform:.4f})")
    print(f"  between-compound sd of alpha (mean)    : {spread:.5f}")
    print(f"  participation ratio of centred alpha   : {participation:.3f} "
          f"of a possible {min(R, len(chem_names) - 1)}")
    print(f"  singular value spectrum (normalised)   : "
          f"{np.round(p[:min(6, len(p))], 4).tolist()}")

    collapsed = participation < 1.5
    verdict = (
        f"The chemical modulation spans an effective {participation:.2f} independent "
        f"directions across {len(chem_names)} compounds. "
        + ("That is close to one, so the attention modulation has largely collapsed to a "
           "single shared departure from static attention: it is chemistry-gated in "
           "magnitude more than in pattern, and the architecture is doing less than the "
           "formula allows."
           if collapsed else
           "The modulation therefore uses genuinely distinct chemical directions rather "
           "than a single shared shift, which is what the cross-attention formulation "
           "is meant to provide."))
    print(f"\n  {verdict}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Mt = torch.from_numpy(Mf)
    probe = [c for c in ["Fluconazole", "Clotrimazole", "Rapamycin", "CHX", "Nocodazole",
                         "Geldanamycin", "Staurosporine", "NaCl", "Sorbitol",
                         "Tunicamycin", "Cisplatin", "Trichostatin A"]
             if c in chem_names]
    code = {c: i for i, c in enumerate(chem_names)}
    idx = torch.tensor([code[c] for c in probe])
    with torch.no_grad():
        _rep, att = net.xattn(net.E, net.nbr_idx, net.nbr_msk, Mt[idx], return_attn=True)
        att = att.mean(-1).numpy()
    dev = att - att.mean(0, keepdims=True)
    per_prot = np.abs(dev).mean(2)

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))

    ax = axes[0]
    im = ax.imshow(A[[code[c] for c in probe]], aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(probe)))
    ax.set_yticklabels(probe, fontsize=7)
    ax.set_xlabel("attention mode r")
    ax.set_title("Mode mixture alpha(M)\nwhich modes each compound selects", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    ax.plot(np.arange(1, len(p) + 1), np.cumsum(p), marker="o", ms=4, color="#4C72B0")
    ax.axhline(0.9, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("component")
    ax.set_ylabel("cumulative variance of alpha")
    ax.set_title(f"Effective rank of the modulation\nparticipation ratio = "
                 f"{participation:.2f}", fontsize=9)
    ax.set_ylim(0, 1.02)

    ax = axes[2]
    im = ax.imshow(per_prot[:, np.argsort(-per_prot.mean(0))[:120]], aspect="auto",
                   cmap="magma")
    ax.set_yticks(range(len(probe)))
    ax.set_yticklabels(probe, fontsize=7)
    ax.set_xlabel("top-120 most chemistry-responsive proteins")
    ax.set_title("Attention deviation from the panel mean\n(baseline weight = 1/12 = 0.083)",
                 fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[3]
    corr = np.corrcoef(per_prot)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(probe)))
    ax.set_xticklabels(probe, rotation=90, fontsize=6)
    ax.set_yticks(range(len(probe)))
    ax.set_yticklabels(probe, fontsize=6)
    ax.set_title("Between-compound similarity of\nattention reweighting", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(FIGURES / "step6_cross_attention_weights.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIGURES / "step6_cross_attention_weights.pdf", bbox_inches="tight")
    plt.close()
    print(" figure -> figures/step6_cross_attention_weights.png")

    rep = {
        "step": "6_3b_attention_rank_diagnostics",
        "checkpoint": str(ckpt_p),
        "n_modes": int(R),
        "n_compounds": len(chem_names),
        "mean_mixture_entropy": float(ent.mean()),
        "uniform_mixture_entropy": ent_uniform,
        "between_compound_alpha_sd": spread,
        "participation_ratio": participation,
        "normalised_singular_spectrum": [float(x) for x in p],
        "max_attention_deviation_from_panel_mean": float(per_prot.max()),
        "baseline_attention_weight": 1.0 / 12.0,
        "modulation_collapsed_to_single_mode": bool(collapsed),
        "verdict": verdict,
        "method_note": (
            "alpha(M) = softmax(MLP(M)) over R modes, averaged across heads. The "
            "participation ratio of the compound-centred alpha matrix's singular "
            "spectrum measures how many independent chemical directions the attention "
            "genuinely responds to; a value near 1 indicates collapse to a single "
            "shared pattern."),
    }
    (RESULTS / "step6_attention_rank_diagnostics.json").write_text(json.dumps(rep, indent=2))
    print(" -> results/step6_attention_rank_diagnostics.json")


if __name__ == "__main__":
    sys.exit(main())
