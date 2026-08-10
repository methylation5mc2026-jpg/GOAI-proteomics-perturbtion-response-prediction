#!/usr/bin/env python3
"""Emit writing_outputs/drafts/step6_tables/*.tex -- generated LaTeX table bodies.

Same contract as the number emitters: any table whose rows are a transcription of an
artefact is generated, never hand-keyed. Each file contains only the rows between
\\midrule and \\bottomrule, so the surrounding table environment (caption, label,
column spec) stays in the manuscript where it can be read in context.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
import pathlib

ROOT = pathlib.REPO_ROOT
RESULTS = ROOT / "results"
OUT = ROOT / "writing_outputs" / "drafts" / "step6_tables"
OUT.mkdir(parents=True, exist_ok=True)

written: list[str] = []
skipped: list[str] = []


def load(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return None


def esc(s) -> str:
    """Escape LaTeX specials in free text drawn from an artefact."""
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&")
            .replace("%", r"\%").replace("$", r"\$").replace("#", r"\#")
            .replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
            .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}"))


def esc_clip(text, n: int) -> str:
    """Escape, then truncate at a word boundary with a LaTeX ellipsis if shortened.

    The ellipsis is appended after escaping, so ``esc`` cannot mangle its backslash.
    """
    t = str(text).strip()
    if len(t) <= n:
        return esc(t)
    cut = t[:n].rsplit(" ", 1)[0].rstrip(" ,;:")
    return esc(cut) + r"\,\ldots"


def sci(v, nd: int = 2) -> str:
    r"""LaTeX scientific notation, e.g. $4.88\times 10^{-6}$.

    "{:.2e}" would render inside math mode as "4.88e - 06", the exponent's sign becoming
    a binary operator.
    """
    txt = f"{float(v):.{nd}e}"
    mant, exp = txt.split("e")
    e = int(exp)
    return f"${mant}$" if e == 0 else rf"${mant}\times 10^{{{e}}}$"


def emit(name: str, rows: list[str]) -> None:
    """Write a table body: rows separated by \\\\, with NO trailing row break.

    The omitted final ``\\\\`` is load-bearing. LaTeX's ``\\input`` appends a ``\\relax``
    after the file's contents; if the file ends with a row break, that ``\\relax`` lands
    at the start of a fresh row and opens a spurious cell, so the following
    ``\\bottomrule`` fails with "Misplaced \\noalign". The manuscript therefore supplies
    the last row break itself, as ``\\input{...} \\\\`` immediately before ``\\bottomrule``.
    """
    if not rows:
        skipped.append(name)
        return
    (OUT / name).write_text(" \\\\\n".join(rows) + "\n", encoding="utf-8")
    written.append(f"{name} ({len(rows)} rows)")


# ==================================================================== 6.1 mechanism val
mval = load("step6_itpv_mechanism_validation.json")
if mval:
    rows = []
    for p in mval.get("positives", []):
        hits = [h for h in p.get("hits", []) if h.get("status") == "annotated"]
        if hits:
            best = max(hits, key=lambda h: float(h.get("pactivity", 0.0)))
            prots = ", ".join(sorted({esc(h["protein"]) for h in hits})[:4])
            detail = f"\\code{{{prots}}}"
            pact = f"${float(best['pactivity']):.2f}$"
            mark = r"\checkmark"
        else:
            missing = ", ".join(sorted({esc(h["protein"]) for h in p.get("hits", [])}))
            detail = f"\\code{{{missing}}} (no annotation)"
            pact = "---"
            mark = r"$\times$"
        rows.append(f"{esc(p.get('compound'))} & {esc_clip(p.get('rationale'), 78)} & "
                    f"{detail} & {pact} & {mark}")
    emit("mechval_positives.tex", rows)

    nrows = []
    for n in mval.get("negative_controls", []):
        hits = n.get("hits", [])
        prots = ", ".join(esc(h.get("protein")) for h in hits)
        worst = max((float(h.get("pactivity", 0.0)) for h in hits), default=0.0)
        ok = r"\checkmark" if n.get("all_zero", worst == 0.0) else r"$\times$"
        nrows.append(f"{esc(n.get('compound'))} & \\code{{{prots}}} & "
                     f"{esc_clip(n.get('rationale', ''), 66)} & ${worst:.2f}$ & {ok}")
    emit("mechval_negatives.tex", nrows)

# ==================================================================== 6.1 top compounds
aff = load("step6_target_affinity_report.json")
if aff:
    pc = [c for c in aff.get("per_compound", []) if c.get("n_yeast_targets", 0) > 0]
    pc.sort(key=lambda c: -c.get("n_yeast_targets", 0))
    rows = []
    for c in pc[:16]:
        rows.append(
            f"{esc(c['chemical'])} & \\code{{{esc(c['chembl_id'])}}} & "
            f"{esc(c['match_method'].replace('_', ' '))} & "
            f"${c['n_chembl_targets']:,}$".replace(",", "{,}") +
            f" & ${c['n_yeast_targets']}$ & ${float(c['max_pactivity']):.2f}$")
    emit("itpv_top_compounds.tex", rows)

    zero = [c["chemical"] for c in aff.get("per_compound", [])
            if c.get("n_yeast_targets", 0) == 0]
    if zero:
        (OUT / "itpv_zero_compounds.tex").write_text(
            ", ".join(esc(z) for z in sorted(zero)) + "\n", encoding="utf-8")
        written.append(f"itpv_zero_compounds.tex ({len(zero)} names)")

# ==================================================================== 6.2 mechanism loss
mech = load("step6_mechanism_loss_report.json")
if mech:
    rows = []
    for name, d in sorted((mech.get("complex_membership") or {}).items(),
                          key=lambda kv: -kv[1].get("n_measured", 0)):
        rows.append(f"\\code{{{esc(name)}}} & ${d.get('n_curated')}$ & "
                    f"${d.get('n_measured')}$")
    emit("mech_complexes.tex", rows)

    iso = mech.get("term_isolation") or {}
    ver = (mech.get("verification") or {}).get("per_term_deviation") or {}
    LABEL = {"mse": "Masked squared error (data term)",
             "pcc_loss": "Per-sample correlation term",
             "complex": "Complex coherence, Eq.~\\ref{eq:complexloss}",
             "flux": "Metabolic flux consistency, Eq.~\\ref{eq:fluxloss}"}
    rows = []
    for key in ("mse", "pcc_loss", "complex", "flux"):
        iso_key = {"mse": "mse_only", "pcc_loss": "pcc_only",
                   "complex": "complex_only", "flux": "flux_only"}[key]
        it = iso.get(iso_key) or {}
        dev = ver.get(key)
        s, e = it.get("start"), it.get("end")
        rows.append(
            f"{LABEL[key]} & "
            + (f"${s:.2f}$" if isinstance(s, float) else "---") + " & "
            + (sci(e) if isinstance(e, float) else "---") + " & "
            + (sci(dev) if isinstance(dev, float) else "---"))
    emit("mech_terms.tex", rows)

# ==================================================================== 6.3 xattn folds
xtr = load("step6_xattn_training.json")
if xtr:
    folds = xtr.get("folds") or {}
    rows = []
    for k in sorted(folds):
        f = folds[k]
        cks = f.get("checkpoints") or {}
        ck_ep = sorted(cks)[0] if cks else None
        ck_pcc = cks[ck_ep]["dev_pcc_at_checkpoint"] if ck_ep else None
        hist = f.get("history") or []
        secs = hist[-1].get("elapsed_s") if hist else None
        dev = [(e["epoch"], e["dev_pcc"]) for e in hist if e.get("dev_pcc") is not None]
        fin = dev[-1][1] if dev else None
        d_fin = (fin - ck_pcc) if isinstance(fin, float) and isinstance(ck_pcc, float) \
            else None
        rows.append(
            f"{esc(k)} & ${f.get('n_dev', 0):,}$".replace(",", "{,}")
            + " & " + (f"${ck_pcc:.4f}$" if isinstance(ck_pcc, float) else "---")
            + " & " + (f"${fin:.4f}$" if isinstance(fin, float) else "---")
            + " & " + (f"${d_fin:+.4f}$" if isinstance(d_fin, float) else "---")
            + " & " + (f"${f['best_dev_pcc']:.4f}$"
                       if isinstance(f.get("best_dev_pcc"), float) else "---")
            + " & " + (f"${f['best_epoch']}$" if f.get("best_epoch") else "---")
            + " & " + (f"${secs:.0f}$" if isinstance(secs, float) else "---"))
    emit("xattn_folds.tex", rows)

# ==================================================================== 6.4 ladder
stack = load("step6_cluster_weights.json")
if stack:
    lad = stack.get("ladder") or {}
    rungs = [(k, v) for k, v in lad.items()
             if isinstance(v, dict) and "oof_total_surrogate" in v]
    rungs.sort(key=lambda kv: (kv[1].get("n_clusters", 0), kv[1].get("alpha", 0.0)))
    sc = stack.get("scalar_rung") or {}
    rows = []
    if sc.get("oof_total") is not None:
        rows.append(f"Scalar ($K=1$) & $1$ & $0.000$ & scalar & "
                    f"${sc['oof_total']:.6f}$ & "
                    f"${4 * len(stack.get('roles', [])) }$ & "
                    f"${sc.get('n_evaluations', 0):,}$".replace(",", "{,}"))
    for k, v in rungs:
        rows.append(
            f"\\code{{{esc(v['cluster_col'])}}} & ${v['n_clusters']}$ & "
            f"${v['alpha']:.3f}$ & \\code{{{esc(v.get('warm_started_from', '--'))}}} & "
            f"${v['oof_total_surrogate']:.6f}$ & "
            f"${4 * len(stack.get('roles', [])) * v['n_clusters']}$ & "
            + f"${v.get('n_evaluations', 0):,}$".replace(",", "{,}"))
    emit("hier_ladder.tex", rows)

    # module decomposition: Step-6 frozen vs Step-5 replay vs benchmark
    MODS = [("m1\\_abundance", "m1_abundance", "0.20"),
            ("m2\\_fold\\_change", "m2_fold_change", "0.25"),
            ("m3\\_s1\\_chem", "m3_s1_chem", "0.20"),
            ("m3\\_s2\\_strain", "m3_s2_strain", "0.20"),
            ("m3\\_s3\\_both", "m3_s3_both", "0.05"),
            ("m3\\_time", "m3_time", "0.05"),
            ("m4\\_dep", "m4_dep", "0.05")]
    scores = stack.get("scores") or {}
    a = (scores.get("val:hier_cluster_oof_frozen") or {}).get("module_scores") or {}
    b = (scores.get("val:step5_frozen_replay") or {}).get("module_scores") or {}
    c = (scores.get("val:bench") or {}).get("module_scores") or {}
    rows = []
    for label, key, w in MODS:
        va, vb, vc = a.get(key), b.get(key), c.get(key)
        diff = (va - vb) if isinstance(va, float) and isinstance(vb, float) else None
        rows.append(
            f"\\code{{{label}}} & ${w}$ & "
            + (f"${vc:.4f}$" if isinstance(vc, float) else "---") + " & "
            + (f"${vb:.4f}$" if isinstance(vb, float) else "---") + " & "
            + (f"${va:.4f}$" if isinstance(va, float) else "---") + " & "
            + (f"${diff:+.4f}$" if isinstance(diff, float) else "---"))
    emit("hier_modules.tex", rows)

    # per-member totals on both cohorts
    ROLE_LABEL = {"gbdt_tab": "GBDT, tabular", "gbdt_mol": "GBDT, 2D molecular",
                  "gbdt_mol3d": "GBDT, 3D molecular", "bench": "Group-mean benchmark",
                  "gnn": "Graph-regularised network", "dl": "Dense MLP-ResNet",
                  "xattn": "Cross-attention transformer"}
    rows = []
    for role in stack.get("roles", []):
        v = (scores.get("val:" + role) or {}).get("total_score")
        rows.append(f"{ROLE_LABEL.get(role, esc(role))} & \\code{{{esc(role)}}} & "
                    + (f"${v:.4f}$" if isinstance(v, float) else "---"))
    cv = (scores.get("val:control_anchor") or {}).get("total_score")
    if isinstance(cv, float):
        rows.append("Control anchor (zero $\\dlt$) & \\code{control\\_anchor} & "
                    f"${cv:.4f}$")
    emit("hier_members.tex", rows)

# ==================================================================== 6.5 scaling
scal = load("step6_compute_scaling_report.json")
if scal:
    rows = []
    for r in scal.get("per_fold", []):
        ck = [k for k in r if k.startswith("dev_pcc_at_epoch_")]
        short = r.get(ck[0]) if ck else None
        ssec = r.get(ck[0].replace("dev_pcc_at_epoch_", "seconds_to_epoch_")) if ck else None
        fin = r.get("dev_pcc_at_final_epoch")
        bst = r.get("best_dev_pcc_over_run")
        d_fin = (fin - short) if isinstance(fin, float) and isinstance(short, float) else None
        d_bst = (bst - short) if isinstance(bst, float) and isinstance(short, float) else None
        rows.append(
            f"{esc(r.get('fold'))} & "
            + (f"${short:.4f}$" if isinstance(short, float) else "---") + " & "
            + (f"${fin:.4f}$" if isinstance(fin, float) else "---") + " & "
            + (f"${d_fin:+.4f}$" if isinstance(d_fin, float) else "---") + " & "
            + (f"${bst:.4f}$" if isinstance(bst, float) else "---") + " & "
            + (f"${r['best_epoch']}$" if r.get("best_epoch") else "---") + " & "
            + (f"${d_bst:+.4f}$" if isinstance(d_bst, float) else "---") + " & "
            + (f"${ssec:.0f}$" if isinstance(ssec, (int, float)) else "---") + " & "
            + (f"${r['wall_clock_seconds']:.0f}$"
               if isinstance(r.get("wall_clock_seconds"), float) else "---"))
    emit("scaling_folds.tex", rows)

# ==================================================================== 6.5 verification
ver = load("step6_verification_report.json")
if ver:
    chk = ((ver.get("test_predictions") or {}).get("artefact_integrity") or {}).get("checks") or []
    rows = []
    for c in chk:
        mark = r"\checkmark" if c.get("passed") else (
            r"$\times$" if c.get("fatal", True) else r"$\circ$")
        kind = "fatal" if c.get("fatal", True) else "diagnostic"
        rows.append(f"{esc(c.get('name'))} & {kind} & {mark} & "
                    f"{{\\footnotesize {esc_clip(c.get('detail', ''), 96)}}}")
    emit("ver_checks.tex", rows)

    crits = ver.get("success_criteria") or ver.get("criteria") or []
    if isinstance(crits, list) and crits:
        rows = []
        for c in crits:
            mark = r"\checkmark" if c.get("met") else r"$\times$"
            rows.append(f"{esc(c.get('name', c.get('criterion', '')))} & "
                        f"{esc(c.get('target', ''))} & {esc(c.get('observed', ''))} & {mark}")
        emit("ver_criteria.tex", rows)

# Emit a tiny macro file describing this emitter's own output, so the manuscript can
# state how many rows it generated without anyone counting them by hand.
_bodies = sorted(q for q in OUT.glob("*.tex")
                 if q.stem not in ("itpv_zero_compounds", "_counts"))
_rows = sum(sum(1 for ln in q.read_text(encoding="utf-8").splitlines() if ln.strip())
            for q in _bodies)
(OUT / "_counts.tex").write_text(
    "% Generated by workflow/41_emit_step6_tables.py -- do not edit by hand.\n"
    f"\\newcommand{{\\SSixNTableBodies}}{{{len(_bodies)}}}\n"
    f"\\newcommand{{\\SSixNTableRows}}{{{_rows}}}\n", encoding="utf-8")

print(f"wrote {len(written)} table bodies into {OUT}")
print(f"  _counts.tex: {len(_bodies)} bodies, {_rows} generated rows")
for w in written:
    print("  " + w)
if skipped:
    print("  empty/absent (artefact not ready): " + ", ".join(skipped))
