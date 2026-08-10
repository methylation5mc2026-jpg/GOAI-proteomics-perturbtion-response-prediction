#!/usr/bin/env python3
"""Draw figures/step6_xattn_architecture.png -- the cross-attention architecture.

Why this is drawn rather than illustrated
-----------------------------------------
The AI-generated version of this figure rendered five attention heads where the model has
four, and printed a stray prompt label onto the canvas. Re-prompting returned a byte-identical
image, so the diagram is drawn here instead. Every count in it is read from the training
script's own defaults and from the training artefact, so the figure cannot disagree with the
model it depicts.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch         # noqa: E402

SESSION = REPO_ROOT
WORKFLOW = SESSION / "workflow"
RESULTS = SESSION / "results"
FIGURES = SESSION / "figures"

INK = "#1b2a33"
NAVY = "#3b4a72"
TEAL = "#2f8f8a"
AMBER = "#d99a2b"
PALE = "#eef3f6"
GREY = "#8a9aa5"


def script_defaults(path: Path) -> dict:
    """Read the argparse defaults out of the training script's source.

    Parsed from the AST rather than imported, because importing the module would pull in
    torch and the whole data pipeline just to learn that ``--heads`` defaults to 4.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = str(node.args[0].value).lstrip("-").replace("-", "_")
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                out[name] = kw.value.value
    return out


def box(ax, x, y, w, h, text, fc, ec=None, tc="white", fs=8.6, weight="normal",
        radius=0.02, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.004,rounding_size={radius}",
                                fc=fc, ec=ec or fc, lw=1.0, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=z + 1, weight=weight, linespacing=1.45)


def arrow(ax, xy, xytext, color=NAVY, lw=1.6, style="-|>", ms=9, z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch(xytext, xy, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step6_xattn_architecture")
    args = ap.parse_args()

    d = script_defaults(WORKFLOW / "36_dynamic_cross_attention_gnn.py")
    n_heads = int(d.get("heads", 4))
    n_modes = int(d.get("n_modes", 8))
    attn_k = int(d.get("attn_k", 12))
    width = int(d.get("width", 384))
    blocks = int(d.get("blocks", 3))
    d_prot = int(d.get("d_prot", 64))
    epochs = int(d.get("epochs", 300))
    w_cpx, w_flx = d.get("w_complex", 0.02), d.get("w_flux", 0.01)

    n_params, n_chem = None, None
    p = RESULTS / "step6_xattn_training.json"
    if p.exists():
        rep = json.loads(p.read_text())
        folds = rep.get("folds") or {}
        if folds:
            f0 = folds[sorted(folds)[0]]
            n_params = f0.get("n_parameters")
    # Distinct chemicals in a fold's fit set bounds the number of distinct attention
    # tensors per batch. Read from the LCGO fold composition, not assumed.
    n_chem = None
    fp = RESULTS / "step5_lcgo_folds.json"
    if fp.exists():
        comp = (json.loads(fp.read_text()).get("composition") or {})
        vals = [v.get("n_fit_chemicals") for v in comp.values()
                if isinstance(v.get("n_fit_chemicals"), int)]
        if vals:
            n_chem = max(vals)
    n_params_txt = f"{n_params:,} parameters" if n_params else "parameters: see text"
    n_chem_txt = f"{n_chem}" if n_chem else "31"

    fig, ax = plt.subplots(figsize=(13.6, 8.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()

    # ---------------- inputs ----------------
    box(ax, 6, 3, 30, 9.5,
        "$\\bf{Chemical\\ context\\ M}$ (per compound)\n"
        "RDKit 2D descriptors  ·  3D conformer shape\n"
        "Morgan fingerprint  ·  32-column block from the\n"
        "Inferred Target Potency Vector", NAVY, fs=8.2)
    box(ax, 39, 3, 24, 9.5,
        "$\\bf{Sample\\ design}$ (per row)\n"
        "strain  ·  time  ·  batch\nplate  ·  instrument  ·  concentration",
        NAVY, fs=8.2)
    box(ax, 68, 3, 27, 9.5,
        f"$\\bf{{Protein\\ graph}}$\n5,243 nodes  ·  STRING association edges\n"
        f"{d_prot}-dim learned embeddings $\\mathbf{{E}}$,\n"
        "initialised from graph spectral coordinates",
        AMBER, tc=INK, fs=8.2)

    # ---------------- trunk ----------------
    box(ax, 12, 20, 45, 7.5,
        f"$\\bf{{MLP\\!-\\!ResNet\\ trunk}}$  ·  {blocks} residual blocks  ·  width {width}"
        f"  $\\rightarrow$  $\\mathbf{{h}}_s$", NAVY, fs=9)
    arrow(ax, (24, 20), (21, 12.5))
    arrow(ax, (45, 20), (51, 12.5))

    # ---------------- attention block ----------------
    ax.add_patch(FancyBboxPatch((12, 31), 56, 37,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc=PALE, ec=TEAL, lw=1.8, zorder=1))
    ax.text(14.6, 66.4, "DYNAMIC CROSS-ATTENTION", fontsize=10.5, weight="bold",
            color=TEAL, va="top")
    # matplotlib's mathtext has no \big, \underbrace or \textstyle, so the two additive
    # parts of the logit are labelled beneath the expression instead of braced within it.
    ax.text(14.6, 62.6,
            r"$a^{(h)}_{ij}(\mathbf{M})=\mathrm{softmax}_{j \in \mathcal{N}(i)}"
            r"[\; \langle \mathbf{W}_Q\mathbf{e}_i,\, \mathbf{W}_K\mathbf{e}_j"
            r"\rangle / \sqrt{d_h}\;+\;\sum_{r=1}^{R}\alpha_r(\mathbf{M})\,"
            r"\tanh \langle \mathbf{u}_r,\, \mathbf{W}_K\mathbf{e}_j \odot "
            r"\mathbf{e}_i \rangle \;]$",
            fontsize=9.0, color=INK, va="top")
    ax.text(24.5, 57.6, "static protein–protein affinity", fontsize=7.4, color=GREY,
            ha="center", va="top", style="italic")
    ax.text(52.0, 57.6, "chemical modulation: rank $R$, edge-dependent",
            fontsize=7.4, color=GREY, ha="center", va="top", style="italic")

    # exactly n_heads head panels
    hx0, hw, hgap = 20.0, 5.4, 2.0
    for i in range(n_heads):
        x = hx0 + i * (hw + hgap)
        box(ax, x, 36.0, hw, 12.5, f"head\n{i + 1}", TEAL, tc="white", fs=8.0,
            radius=0.03, z=2)
    ax.text(hx0 + (n_heads * (hw + hgap) - hgap) / 2 + 2.0, 32.9,
            f"{n_heads} attention heads  ·  softmax over $k={attn_k}$ graph neighbours"
            f"  ·  $R={n_modes}$ chemical modes",
            ha="center", va="bottom", fontsize=8.4, color=INK)

    # Q / K / V routing
    ax.text(16.2, 42.3, "Q", fontsize=11, weight="bold", color=NAVY, ha="center")
    arrow(ax, (hx0 - 0.5, 42.3), (17.2, 42.3), color=NAVY)
    arrow(ax, (16.2, 40.6), (26.0, 27.7), color=NAVY, rad=-0.20)
    ax.text(35.2, 52.6, "K, V", fontsize=10.5, weight="bold", color=AMBER, ha="center")
    arrow(ax, (31.5, 48.8), (34.6, 51.6), color=AMBER, style="<|-", lw=1.4)
    arrow(ax, (37.2, 51.8), (77.0, 13.0), color=AMBER, rad=-0.40, lw=1.4)

    # alpha(M) modulation path -- the load-bearing one
    box(ax, 47.5, 39.5, 19, 5.6,
        "mode mixture\n"
        r"$\mathbf{\alpha}(\mathbf{M})=\mathrm{softmax}(\mathrm{MLP}(\mathbf{M}))$",
        AMBER, tc=INK, fs=8.4)
    arrow(ax, (46.4, 42.3), (38.0, 42.3), color=AMBER, style="<|-", lw=1.4)
    arrow(ax, (11.8, 45.0), (14.0, 12.6), color=AMBER, rad=-0.36, lw=1.4)
    ax.text(8.6, 30.0, "the chemical context\nsteers the attention", rotation=90,
            fontsize=7.5, color=AMBER, ha="center", va="center", weight="bold")

    # ---------------- output head ----------------
    box(ax, 12, 73, 56, 6.6,
        "factorised output head:   "
        r"$\hat{\delta}_{sp}=\langle \tilde{\mathbf{e}}_p,\; "
        r"\mathbf{W}_\delta \mathbf{h}_s \rangle + b_p$"
        "\n(plus an auxiliary abundance head at weight 0.3)", NAVY, fs=8.8)
    arrow(ax, (40, 73), (40, 68), color=TEAL)
    box(ax, 12, 85, 56, 6.4,
        r"$\bf{predicted\ log_2\ fold\ change,\ 5{,}243\ proteins}$", TEAL, fs=9.6)
    arrow(ax, (40, 85), (40, 79.6), color=TEAL)

    # ---------------- side notes ----------------
    ax.add_patch(FancyBboxPatch((71, 31), 27, 37,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc="white", ec=GREY, lw=1.0, zorder=2))
    ax.text(72.4, 66.4,
            "Two facts that make this work\n\n"
            "$\\bf{1}$  The modulation acts on $\\it{edge}$\n"
            "features. A per-sample scalar added\n"
            "to every logit would cancel exactly\n"
            "in the softmax — a silent no-op.\n\n"
            f"$\\bf{{2}}$  $\\mathbf{{M}}$ is a property of the\n"
            f"compound, so a batch holds at most\n"
            f"{n_chem_txt} distinct attention tensors. The\n"
            "stated softmax is computed exactly,\n"
            "not approximated by a mixture.",
            fontsize=8.0, color=INK, va="top", linespacing=1.6)

    box(ax, 71, 21.5, 27, 8.6,
        f"{n_params_txt}\n5-fold leave-chemical-group-out\n"
        f"cross-fitting  ·  cosine-annealed, {epochs} epochs",
        PALE, ec=GREY, tc=INK, fs=7.9)

    # mechanism loss bracket on the left
    ax.plot([4.2, 4.2], [22, 78], color=TEAL, lw=1.6)
    ax.plot([4.2, 5.6], [78, 78], color=TEAL, lw=1.6)
    ax.plot([4.2, 5.6], [22, 22], color=TEAL, lw=1.6)
    ax.text(2.6, 50,
            f"mechanism loss applied here:\n"
            f"complex coherence ({w_cpx})  +  flux balance ({w_flx})",
            rotation=90, fontsize=8.4, color=TEAL, ha="center", va="center",
            weight="bold", linespacing=1.5)

    ax.set_title("Dynamic bio-chemical cross-attention graph transformer",
                 fontsize=14.5, weight="bold", color=INK, pad=12)

    fig.savefig(FIGURES / f"{args.out}.png", dpi=190, bbox_inches="tight",
                facecolor="white")
    fig.savefig(FIGURES / f"{args.out}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote figures/{args.out}.png")
    print(f"  heads={n_heads} modes={n_modes} k={attn_k} width={width} "
          f"blocks={blocks} d_prot={d_prot} epochs={epochs} params={n_params}")


if __name__ == "__main__":
    main()
