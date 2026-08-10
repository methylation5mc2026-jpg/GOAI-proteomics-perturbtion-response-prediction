#!/usr/bin/env python
"""Step 6.3 -- Dynamic Bio-Chemical Cross-Attention Graph Transformer.

Chemical context (3D pharmacophore PCs, SASA, ITPV target-affinity block)
dynamically modulates the attention weights the model places on STRING PPI
edges:

    Edge_Weight_ij(M) = Softmax_j[ (Q_i W_Q)(K_j W_K)^T / sqrt(d)
                                   + sigma(M . W_M)_ij ]

Two design points worth stating explicitly
------------------------------------------
1. **The modulation must be edge-dependent or it is a no-op.** Softmax is
   invariant to a constant shift of its logits. If ``sigma(M . W_M)`` produced a
   single per-sample (or per-head) scalar added to every logit in the
   neighbourhood, it would cancel exactly in the softmax and the "dynamic"
   attention would reduce to the static one. We therefore project the chemical
   context onto a learned basis of *edge* features:

       bias_ij(M) = sum_r alpha_r(M) * tanh(<u_r, E_i (*) E_j>)
       alpha(M)   = softmax(MLP(M))

   which is a rank-R realisation of ``sigma(M . W_M)`` over edges, and does
   genuinely change the attention distribution.

2. **The exact per-sample softmax is affordable here.** M is a property of the
   *compound*, not of the individual row, so within any batch there are at most
   as many distinct attention tensors as there are distinct chemicals -- at most
   37 in training, usually far fewer. We compute the attention exactly once per
   distinct chemical in the batch and gather, rather than approximating with a
   post-softmax mixture. No approximation of the stated formula is made.

Training follows the established Step-5 protocol: the 5 LCGO folds, the
``SampleFeaturizer`` fitted per fold on fit rows only, cosine-annealed learning
rate, and cross-fitted OOF predictions written to the shared cache under the
role tag ``xattn``. The mechanism loss from Step 6.2 supplies the complex
co-response and metabolic flux-conservation terms.

Outputs
-------
data/step5_cache/{oof,val,test}_xattn.npy
results/step6_xattn_training.json
figures/step6_cross_attention_weights.png
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

SESSION = REPO_ROOT
WORKFLOW = SESSION / "workflow"
DATA = SESSION / "data"
RESULTS = SESSION / "results"
FIGURES = SESSION / "figures"
CACHE5 = DATA / "step5_cache"
MODELS5 = WORKFLOW / "models_step5"
SEED = 42
ROLE_TAG = "xattn"
sys.path.insert(0, str(WORKFLOW))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Chemical context matrix M
# ---------------------------------------------------------------------------
def build_chem_context(chem_names: list[str], fit_chems: set[str]) -> tuple[np.ndarray, list[str]]:
    """Assemble the per-compound chemical context matrix M.

    Columns: 3D pharmacophore PCs + shape descriptors + SASA (Step 5) and the
    ITPV target-affinity block (Step 6.1). Standardisation uses fit-cohort
    compounds only, so a held-out compound cannot influence its own scaling.
    """
    mol3d = pd.read_parquet(DATA / "step5_mol3d_features.parquet")
    blocks = [mol3d]
    tf = DATA / "step6_target_features.parquet"
    if tf.exists():
        t = pd.read_parquet(tf).set_index("chemical")
        blocks.append(t)
        log(f"  target-affinity block: {t.shape[1]} columns from Step 6.1")
    else:
        log("  !! step6_target_features.parquet absent; running without the ITPV block")
    M = pd.concat(blocks, axis=1)
    M = M.reindex(chem_names)
    cols = list(M.columns)
    A = M.to_numpy(dtype=np.float64)
    A = np.where(np.isfinite(A), A, np.nan)

    fit_rows = np.array([c in fit_chems for c in chem_names])
    if fit_rows.sum() < 2:
        fit_rows = np.ones(len(chem_names), dtype=bool)
    with np.errstate(all="ignore"):
        mu = np.nanmean(A[fit_rows], axis=0)
        sd = np.nanstd(A[fit_rows], axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    A = np.where(np.isfinite(A), A, mu[None, :])
    Z = (A - mu[None, :]) / sd[None, :]
    return np.clip(Z, -8, 8).astype(np.float32), cols


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DynamicCrossAttention(nn.Module):
    """Chemistry-modulated multi-head attention over STRING neighbourhoods."""

    def __init__(self, d_prot: int, d_chem: int, n_heads: int = 4, n_modes: int = 8):
        super().__init__()
        self.h, self.dh, self.R = n_heads, d_prot // n_heads, n_modes
        self.q = nn.Linear(d_prot, d_prot, bias=False)
        self.k = nn.Linear(d_prot, d_prot, bias=False)
        self.v = nn.Linear(d_prot, d_prot, bias=False)
        self.o = nn.Linear(d_prot, d_prot)
        self.norm = nn.LayerNorm(d_prot)
        # W_M: chemical context -> mode mixture alpha(M)
        self.W_M = nn.Sequential(nn.Linear(d_chem, 64), nn.GELU(),
                                 nn.Linear(64, n_modes * n_heads))
        # u_r: learned edge-feature directions, one per (mode, head)
        self.u = nn.Parameter(torch.randn(n_modes, n_heads, self.dh) * 0.02)
        # Zero-init output projection: the model starts as the no-attention
        # baseline and must earn any graph mixing it uses (Step-5 convention).
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)
        # Zero-init the last modulation layer: training starts from static
        # attention, so any dynamic behaviour is a learned departure from it.
        nn.init.zeros_(self.W_M[-1].weight)
        nn.init.zeros_(self.W_M[-1].bias)

    def forward(self, E: torch.Tensor, idx: torch.Tensor, msk: torch.Tensor,
                Mu: torch.Tensor, return_attn: bool = False):
        """E (P,d); idx/msk (P,k); Mu (U,d_chem) -> (U,P,d) protein representations.

        One representation is produced per distinct chemical context row in
        ``Mu``; the caller gathers per sample.
        """
        P, d = E.shape
        U = Mu.shape[0]
        q = self.q(E).reshape(P, self.h, self.dh)
        kk = self.k(E)[idx].reshape(P, -1, self.h, self.dh)      # (P,k,h,dh)
        vv = self.v(E)[idx].reshape(P, -1, self.h, self.dh)
        static = (q[:, None] * kk).sum(-1) / (self.dh ** 0.5)    # (P,k,h)

        # ---- edge features and the chemistry-driven modulation ----
        ef = E[:, None, :].expand(-1, kk.shape[1], -1).reshape(P, -1, self.h, self.dh) * kk
        # b[r,h] over edges: tanh(<u_r_h, E_i (*) E_j>)
        b = torch.tanh(torch.einsum("pkhd,rhd->pkhr", ef, self.u))       # (P,k,h,R)
        alpha = self.W_M(Mu).reshape(U, self.R, self.h)
        alpha = torch.softmax(alpha, dim=1)                              # (U,R,h)
        bias = torch.einsum("pkhr,urh->upkh", b, alpha)                  # (U,P,k,h)

        logits = static[None] + bias                                     # (U,P,k,h)
        logits = logits.masked_fill(msk[None, :, :, None] < 0.5, float("-inf"))
        att = torch.softmax(logits, dim=2)
        att = torch.nan_to_num(att, nan=0.0)                             # isolated proteins
        out = torch.einsum("upkh,pkhd->uphd", att, vv).reshape(U, P, d)
        rep = self.norm(E[None] + self.o(out))
        if return_attn:
            return rep, att
        return rep


class XAttnNet(nn.Module):
    """Sample trunk + chemistry-modulated protein graph transformer head."""

    def __init__(self, fz, emb0: np.ndarray, nbr_idx, nbr_msk, d_chem: int,
                 width: int = 512, n_blocks: int = 3, emb_dim: int = 16,
                 d_prot: int = 64, p_drop: float = 0.25, n_heads: int = 4,
                 n_modes: int = 8):
        super().__init__()

        class ResBlock(nn.Module):
            def __init__(self, dd, p):
                super().__init__()
                self.net = nn.Sequential(nn.LayerNorm(dd), nn.Linear(dd, dd), nn.GELU(),
                                         nn.Dropout(p), nn.Linear(dd, dd))

            def forward(self, x):
                return x + self.net(x)

        self.embs = nn.ModuleList([nn.Embedding(n, emb_dim) for n in fz.cat_sizes()])
        self.ctrl_proj = nn.Sequential(nn.Linear(fz.p, 256), nn.GELU(), nn.Dropout(p_drop))
        self.fp_proj = nn.Sequential(nn.Linear(fz.n_fp, 128), nn.GELU(), nn.Dropout(p_drop))
        d_in = 256 + 128 + fz.n_desc + fz.n_num + emb_dim * len(fz.cat_sizes())
        self.stem = nn.Sequential(nn.Linear(d_in, width), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(width, p_drop) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(width)

        e0 = np.asarray(emb0[:, :d_prot], dtype="float32")
        if e0.shape[1] < d_prot:
            e0 = np.pad(e0, ((0, 0), (0, d_prot - e0.shape[1])))
        rms = float(np.sqrt((e0 ** 2).mean())) or 1.0
        self.E = nn.Parameter(torch.from_numpy(e0 / rms * 0.5))
        self.register_buffer("nbr_idx", torch.from_numpy(nbr_idx))
        self.register_buffer("nbr_msk", torch.from_numpy(nbr_msk))
        self.xattn = DynamicCrossAttention(d_prot, d_chem, n_heads, n_modes)

        self.proj_delta = nn.Linear(width, d_prot)
        self.proj_abs = nn.Linear(width, d_prot)
        self.bias_delta = nn.Parameter(torch.zeros(fz.p))
        self.bias_abs = nn.Parameter(torch.zeros(fz.p))
        nn.init.zeros_(self.proj_delta.weight)
        nn.init.zeros_(self.proj_delta.bias)

    def forward(self, desc, fp, cats, num, ctrl, Mu, inv):
        """``Mu`` (U,d_chem) unique chemical contexts; ``inv`` (B,) row -> U index."""
        e = [emb(cats[:, i]) for i, emb in enumerate(self.embs)]
        h = torch.cat([self.ctrl_proj(ctrl), self.fp_proj(fp), desc, num] + e, dim=1)
        h = self.norm(self.blocks(self.stem(h)))
        Prep = self.xattn(self.E, self.nbr_idx, self.nbr_msk, Mu)       # (U,P,d)
        Pb = Prep[inv]                                                  # (B,P,d)
        d_hat = torch.bmm(Pb, self.proj_delta(h)[:, :, None]).squeeze(-1) + self.bias_delta
        a_hat = torch.bmm(Pb, self.proj_abs(h)[:, :, None]).squeeze(-1) + self.bias_abs
        return d_hat, a_hat


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_xattn(model, fz, meta, C, D, Y, Mchem, chem_code, fit_idx, dev_idx,
                n_epochs, device, mech_loss, lr=2e-3, wd=1e-2, batch=48,
                w_abs=0.3, w_pcc=1.0, label="xattn", eval_every=5,
                checkpoint_epochs=(), seed=SEED):
    """Cosine-annealed training with the Step-6.2 mechanism loss.

    ``checkpoint_epochs`` records a deep copy of the weights at the given epochs,
    which the compute-scaling assessment uses to compare partial-budget training
    against the full run without retraining.
    """
    dl = sys.modules["dl24"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)

    feats = fz.transform(meta, C, fit_idx)
    T = {k: torch.from_numpy(v) for k, v in feats.items()}
    Dt = torch.from_numpy(np.nan_to_num(D[fit_idx], nan=0.0))
    Dm = torch.from_numpy(np.isfinite(D[fit_idx]).astype("float32"))
    Yt = torch.from_numpy(np.nan_to_num(Y[fit_idx], nan=0.0))
    Ym = torch.from_numpy(np.isfinite(Y[fit_idx]).astype("float32"))
    code_fit = torch.from_numpy(chem_code[fit_idx])
    Mt = torch.from_numpy(Mchem).to(device)

    dev_pack = None
    if dev_idx is not None and len(dev_idx):
        df = fz.transform(meta, C, dev_idx)
        dev_pack = ({k: torch.from_numpy(v).to(device) for k, v in df.items()},
                    torch.from_numpy(np.nan_to_num(D[dev_idx], nan=0.0)).to(device),
                    torch.from_numpy(np.isfinite(D[dev_idx]).astype("float32")).to(device),
                    torch.from_numpy(chem_code[dev_idx]).to(device))

    unseen_ids = torch.tensor([fz.vocab[c][dl.UNSEEN] for c in dl.CAT_EMB], dtype=torch.long)
    drop_p = torch.tensor([dl.ENTITY_DROPOUT[c] for c in dl.CAT_EMB], dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    spe = max(1, int(np.ceil(len(fit_idx) / batch)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs * spe, eta_min=lr * 0.02)

    rng = np.random.default_rng(seed)
    hist, ckpts = [], {}
    best = {"epoch": -1, "dev_pcc": -np.inf, "state": None}
    t0 = time.time()

    for ep in range(1, n_epochs + 1):
        model.train()
        order = rng.permutation(len(fit_idx))
        tot = n_b = 0.0
        term_acc = {"mse": 0.0, "pcc_loss": 0.0, "complex": 0.0, "flux": 0.0}
        for s in range(0, len(order), batch):
            sel = torch.from_numpy(order[s: s + batch])
            cats = T["cats"][sel].clone()
            m = torch.rand(cats.shape) < drop_p[None, :]
            cats = torch.where(m, unseen_ids[None, :].expand_as(cats), cats)

            desc, fp = T["desc"][sel].to(device), T["fp"][sel].to(device)
            num, ctrl = T["num"][sel].to(device), T["ctrl"][sel].to(device)
            cats = cats.to(device)
            dt, dm = Dt[sel].to(device), Dm[sel].to(device)
            yt, ym = Yt[sel].to(device), Ym[sel].to(device)
            # Exact dynamic attention: one attention tensor per distinct chemical
            # present in this batch, gathered back to rows.
            uq, inv = torch.unique(code_fit[sel], return_inverse=True)
            Mu = Mt[uq.to(device)]

            pd_, pa_ = model(desc, fp, cats, num, ctrl, Mu, inv.to(device))
            l_mech, terms = mech_loss(pd_, dt, dm)
            l_abs = dl.masked_mse(pa_, yt, ym)
            loss = l_mech + w_abs * l_abs

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            n_b += 1
            for k in term_acc:
                term_acc[k] += terms[k]

        rec = {"epoch": ep, "train_loss": tot / max(1, n_b),
               "lr": float(opt.param_groups[0]["lr"]),
               "elapsed_s": round(time.time() - t0, 2),
               **{f"term_{k}": v / max(1, n_b) for k, v in term_acc.items()}}
        if dev_pack is not None and (ep % eval_every == 0 or ep == n_epochs):
            model.eval()
            with torch.no_grad():
                f, dt, dm, cc = dev_pack
                preds = []
                for s in range(0, len(dev_idx), 128):
                    sl = slice(s, s + 128)
                    uq, inv = torch.unique(cc[sl], return_inverse=True)
                    p_, _ = model(f["desc"][sl], f["fp"][sl], f["cats"][sl],
                                  f["num"][sl], f["ctrl"][sl], Mt[uq], inv)
                    preds.append(p_)
                p_ = torch.cat(preds, 0)
                rec["dev_pcc"] = float(1.0 - dl.neg_pearson_per_sample(p_, dt, dm))
                rec["dev_mse"] = float(dl.masked_mse(p_, dt, dm))
            if rec["dev_pcc"] > best["dev_pcc"]:
                best = {"epoch": ep, "dev_pcc": rec["dev_pcc"],
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
        if ep in checkpoint_epochs:
            ckpts[ep] = {"state": {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()},
                         "elapsed_s": time.time() - t0,
                         "dev_pcc_running_best": best["dev_pcc"],
                         "best_epoch_so_far": best["epoch"]}
            log(f"  [{label}] checkpoint at epoch {ep} "
                f"(best dev PCC so far {best['dev_pcc']:.4f} @ epoch {best['epoch']})")
        hist.append(rec)
        if ep == 1 or ep % 10 == 0 or ep == n_epochs:
            extra = f" dev_pcc={rec['dev_pcc']:.4f}" if "dev_pcc" in rec else ""
            log(f"  [{label}] epoch {ep}/{n_epochs} loss={rec['train_loss']:.4f}{extra}"
                f" | {time.time() - t0:.0f}s")

    if best["state"] is None:
        best = {"epoch": n_epochs, "dev_pcc": float("nan"),
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    return hist, best, ckpts


def xattn_predict(model, fz, meta, C, idx, Mchem, chem_code, device, batch=96):
    model = model.to(device).eval()
    Mt = torch.from_numpy(Mchem).to(device)
    out = np.empty((len(idx), fz.p), dtype="float32")
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            sl = idx[s: s + batch]
            f = fz.transform(meta, C, sl)
            t = {k: torch.from_numpy(v).to(device) for k, v in f.items()}
            cc = torch.from_numpy(chem_code[sl]).to(device)
            uq, inv = torch.unique(cc, return_inverse=True)
            p_, _ = model(t["desc"], t["fp"], t["cats"], t["num"], t["ctrl"],
                          Mt[uq], inv)
            out[s: s + len(sl)] = p_.cpu().numpy()
            if s % (batch * 10) == 0:
                print(f"      predict {s}/{len(idx)}", flush=True)
    return out


# ---------------------------------------------------------------------------
def attention_figure(model, chem_names, code_of_name, Mchem, proteins, device,
                     mech_struct) -> dict:
    """Visualise how chemistry reshapes the attention placed on PPI edges."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = model.to(device).eval()
    Mt = torch.from_numpy(Mchem).to(device)
    probe = [c for c in ["Fluconazole", "Clotrimazole", "Rapamycin", "CHX",
                         "Nocodazole", "Geldanamycin", "Staurosporine", "NaCl",
                         "Sorbitol", "H2O2", "Tunicamycin", "Cisplatin"]
             if c in code_of_name]
    if len(probe) < 2:
        probe = chem_names[:8]
    codes = torch.tensor([code_of_name[c] for c in probe], device=device)
    with torch.no_grad():
        _rep, att = model.xattn(model.E, model.nbr_idx, model.nbr_msk,
                                Mt[codes], return_attn=True)
        att = att.mean(-1).cpu().numpy()          # (U,P,k) mean over heads

    # Per-compound deviation from the panel-mean attention pattern.
    dev = att - att.mean(axis=0, keepdims=True)
    per_prot = np.abs(dev).mean(axis=2)           # (U,P)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    im = ax.imshow(per_prot[:, np.argsort(-per_prot.mean(0))[:120]],
                   aspect="auto", cmap="magma")
    ax.set_yticks(range(len(probe)))
    ax.set_yticklabels(probe, fontsize=7)
    ax.set_xlabel("top-120 most chemistry-responsive proteins")
    ax.set_title("Chemistry-driven attention deviation\n(|dev| from panel mean)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax = axes[1]
    spread = per_prot.mean(axis=1)
    ax.barh(range(len(probe)), spread, color="#4C72B0")
    ax.set_yticks(range(len(probe)))
    ax.set_yticklabels(probe, fontsize=7)
    ax.set_xlabel("mean |attention deviation|")
    ax.set_title("How far each compound moves the\nattention from the panel mean", fontsize=9)

    ax = axes[2]
    ei = mech_struct["edge_i"]
    sel = np.argsort(-per_prot.mean(0))[:40]
    corr = np.corrcoef(per_prot[:, sel])
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(probe)))
    ax.set_xticklabels(probe, rotation=90, fontsize=6)
    ax.set_yticks(range(len(probe)))
    ax.set_yticklabels(probe, fontsize=6)
    ax.set_title("Between-compound similarity of\nattention reweighting", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(FIGURES / "step6_cross_attention_weights.png", dpi=200,
                bbox_inches="tight")
    plt.savefig(FIGURES / "step6_cross_attention_weights.pdf", bbox_inches="tight")
    plt.close()
    log("  figure -> figures/step6_cross_attention_weights.png")
    return {"probe_compounds": probe,
            "mean_abs_attention_deviation": {c: float(s) for c, s in zip(probe, spread)},
            "max_protein_deviation": float(per_prot.max()),
            "note": ("attention deviation is measured against the mean attention "
                     "pattern across the probed compounds; a strictly zero map "
                     "would mean the chemical modulation was never engaged")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--d-prot", type=int, default=64)
    ap.add_argument("--attn-k", type=int, default=12)
    ap.add_argument("--n-modes", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--w-complex", type=float, default=0.02)
    ap.add_argument("--w-flux", type=float, default=0.01)
    ap.add_argument("--checkpoints", default="150")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("=" * 78)
    log("Step 6.3  Dynamic Bio-Chemical Cross-Attention Graph Transformer")
    log("=" * 78)
    log(f"torch {torch.__version__} device={device} epochs={args.epochs}")
    MODELS5.mkdir(parents=True, exist_ok=True)

    dl = load_module(WORKFLOW / "24_train_deep_learning.py", "dl24")
    st30 = load_module(WORKFLOW / "30_gnn_and_cluster_stacking.py", "st30")
    lc = load_module(WORKFLOW / "29_lcgo_oof_matrix.py", "lcgo29")
    mech = load_module(WORKFLOW / "35_metabolic_mechanism_loss.py", "mech35")
    S4 = sys.modules["step4_common"]

    G = st30.load_graph()
    adj = G["adj"]
    nbr_idx, nbr_msk = st30.topk_neighbours(adj, k=args.attn_k)
    log(f"  graph: {G['graph_source']}, {adj.nnz // 2} undirected edges")

    ctx = S4.load_context()
    meta, Y, D, C = ctx["meta"], ctx["Y"], ctx["D"], ctx["C"]
    masks, VS = ctx["masks"], ctx["VS"]
    proteins = ctx["proteins"]
    train_mask = masks[VS.TRAIN_SPLIT]
    folds = lc.build_folds(meta, train_mask, seed=SEED)
    tr_idx = np.flatnonzero(train_mask)
    pos_of_row = -np.ones(len(meta), dtype=np.int64)
    pos_of_row[tr_idx] = np.arange(len(tr_idx))

    struct = mech.build_structures(proteins)
    log(f"  mechanism structures: {len(struct['edge_i'])} complex edges, "
        f"S={struct['S'].shape}")

    # --- chemical context, one row per distinct compound label ---
    chem_all = meta[S4.CHEM_COL].astype(str).to_numpy()
    chem_names = sorted(set(chem_all))
    code_of_name = {c: i for i, c in enumerate(chem_names)}
    chem_code = np.array([code_of_name[c] for c in chem_all], dtype=np.int64)
    fit_chems_global = set(meta.loc[train_mask, S4.CHEM_COL].astype(str))
    Mchem, mcols = build_chem_context(chem_names, fit_chems_global)
    log(f"  chemical context M: {Mchem.shape} over {len(chem_names)} compounds")

    ckpt_eps = tuple(int(x) for x in args.checkpoints.split(",") if x)
    want_folds = [int(x) for x in args.folds.split(",") if x != ""]
    report = {"step": "6_3_dynamic_cross_attention", "seed": SEED, "device": device,
              "role_tag": ROLE_TAG, "epochs": args.epochs, "attn_k": args.attn_k,
              "n_modes": args.n_modes, "n_heads": args.heads, "batch": args.batch,
              "chem_context_dim": int(Mchem.shape[1]),
              "chem_context_columns": mcols,
              "mechanism_loss_coefficients": {"w_mse": 1.0, "w_pcc": 0.5,
                                              "w_complex": args.w_complex,
                                              "w_flux": args.w_flux},
              "architecture_note": (
                  "Softmax is shift invariant, so a chemical modulation that added a "
                  "single scalar per sample would cancel exactly and leave the "
                  "attention static. The modulation is therefore projected onto a "
                  "learned rank-R basis of edge features, giving a genuinely "
                  "edge-dependent bias. Because the chemical context is a property of "
                  "the compound rather than the row, the exact per-sample softmax is "
                  "computed once per distinct chemical in each batch -- no "
                  "approximation of the stated formula is used."),
              "folds": {}}

    oof = np.full((len(tr_idx), Y.shape[1]), np.nan, dtype="float32")
    n_epochs = 3 if args.smoke else args.epochs
    for f in want_folds:
        fit_mask = folds["fits"][f]
        dev_rows = np.flatnonzero(folds["fold"] == f)
        log(f"\n===== xattn fold {f}: fit n={int(fit_mask.sum())} "
            f"dev n={len(dev_rows)} =====")
        fz = dl.SampleFeaturizer(meta, C, Y, fit_mask)
        fit_chems = set(meta.loc[fit_mask, S4.CHEM_COL].astype(str))
        Mf, _ = build_chem_context(chem_names, fit_chems)
        net = XAttnNet(fz, G["embedding"], nbr_idx, nbr_msk, Mf.shape[1],
                       width=args.width, n_blocks=args.blocks, d_prot=args.d_prot,
                       n_heads=args.heads, n_modes=args.n_modes)
        n_par = sum(p.numel() for p in net.parameters())
        log(f"  {n_par:,} parameters")
        ml = mech.MechanismLoss(struct, w_mse=1.0, w_pcc=0.5,
                                w_complex=args.w_complex, w_flux=args.w_flux,
                                device=device)
        hist, best, ckpts = train_xattn(
            net, fz, meta, C, D, Y, Mf, chem_code, np.flatnonzero(fit_mask),
            dev_rows, n_epochs, device, ml, lr=args.lr, batch=args.batch,
            label=f"xattn_f{f}", eval_every=args.eval_every,
            checkpoint_epochs=ckpt_eps)
        net.load_state_dict(best["state"])
        oof[pos_of_row[dev_rows]] = xattn_predict(net, fz, meta, C, dev_rows,
                                                  Mf, chem_code, device)
        torch.save(best["state"], MODELS5 / f"{ROLE_TAG}_fold{f}.pt")
        rec = {"n_fit": int(fit_mask.sum()), "n_dev": int(len(dev_rows)),
               "n_parameters": int(n_par), "best_epoch": best["epoch"],
               "best_dev_pcc": best["dev_pcc"], "history": hist,
               "seconds": round(hist[-1]["elapsed_s"], 1)}
        # Compute-scaling evidence: score the partial-budget checkpoints too.
        for ep, ck in ckpts.items():
            net.load_state_dict(ck["state"])
            p = xattn_predict(net, fz, meta, C, dev_rows, Mf, chem_code, device)
            dt = torch.from_numpy(np.nan_to_num(D[dev_rows], nan=0.0))
            dm = torch.from_numpy(np.isfinite(D[dev_rows]).astype("float32"))
            pcc = float(1.0 - dl.neg_pearson_per_sample(torch.from_numpy(p), dt, dm))
            rec.setdefault("checkpoints", {})[str(ep)] = {
                "epoch": ep, "dev_pcc_at_checkpoint": pcc,
                "elapsed_s": round(ck["elapsed_s"], 1)}
            log(f"  checkpoint epoch {ep}: dev per-sample PCC {pcc:.4f} "
                f"({ck['elapsed_s']:.0f}s)")
        net.load_state_dict(best["state"])
        report["folds"][f"fold{f}"] = rec
        log(f"  fold {f}: best epoch {best['epoch']} dev PCC {best['dev_pcc']:.4f}")
        if f == want_folds[0] and not args.smoke:
            report["attention_diagnostics"] = attention_figure(
                net, chem_names, code_of_name, Mf, proteins, device, struct)
        del net, fz
        torch.cuda.empty_cache()

    if args.smoke:
        got = oof[pos_of_row[np.flatnonzero(folds["fold"] == want_folds[0])]]
        log(f"\nSMOKE: fold predictions {got.shape} finite={bool(np.isfinite(got).all())} "
            f"sd={float(got.std()):.4f} range [{float(got.min()):+.3f}, {float(got.max()):+.3f}]")
        if not np.isfinite(got).all():
            raise AssertionError("non-finite predictions")
        log("=== smoke OK (no artefact written) ===")
        return

    n_complete = int(np.isfinite(oof).all(axis=1).sum())
    log(f"\nxattn OOF: {n_complete}/{len(tr_idx)} rows complete "
        f"({100 * n_complete / len(tr_idx):.1f}%)")
    st30.cache5_put(f"oof_{ROLE_TAG}", np.nan_to_num(oof, nan=0.0))
    report["oof_rows_complete"] = n_complete
    report["oof_coverage_fraction"] = n_complete / len(tr_idx)

    # ---- refit on all train rows for val / test ----
    log("\n===== xattn refit on all train rows =====")
    n_ep2 = max(5, int(np.median([report["folds"][k]["best_epoch"]
                                  for k in report["folds"]])))
    log(f"  epochs from the fold-wise selections (median): {n_ep2}")
    fz_tr = dl.SampleFeaturizer(meta, C, Y, train_mask)
    Mtr, _ = build_chem_context(chem_names, fit_chems_global)
    net = XAttnNet(fz_tr, G["embedding"], nbr_idx, nbr_msk, Mtr.shape[1],
                   width=args.width, n_blocks=args.blocks, d_prot=args.d_prot,
                   n_heads=args.heads, n_modes=args.n_modes)
    ml = mech.MechanismLoss(struct, w_mse=1.0, w_pcc=0.5, w_complex=args.w_complex,
                            w_flux=args.w_flux, device=device)
    hist2, best2, _ = train_xattn(net, fz_tr, meta, C, D, Y, Mtr, chem_code, tr_idx,
                                  None, n_ep2, device, ml, lr=args.lr,
                                  batch=args.batch, label="xattn_refit",
                                  eval_every=args.eval_every)
    net.load_state_dict(best2["state"])
    torch.save(best2["state"], MODELS5 / f"{ROLE_TAG}_refit.pt")
    report["refit"] = {"n_epochs": n_ep2, "history": hist2}

    val_idx = np.flatnonzero(masks["all_val"])
    st30.cache5_put(f"val_{ROLE_TAG}",
                    xattn_predict(net, fz_tr, meta, C, val_idx, Mtr, chem_code, device))

    log("predicting the test cohort ...")
    te = ctx["S3"].load_test(ctx["proteins"], ctx["meta_all"], ctx["M_all"])
    te_chem = te["meta"][S4.CHEM_COL].astype(str).to_numpy()
    unknown = sorted(set(te_chem) - set(code_of_name))
    if unknown:
        # A test compound never seen in train_val has no context row. Extend the
        # context matrix rather than dropping the row, so every test sample is
        # predicted; the new rows are built from the same descriptor sources.
        log(f"  {len(unknown)} test compounds absent from train_val metadata: {unknown}")
        ext_names = chem_names + unknown
        Mext, _ = build_chem_context(ext_names, fit_chems_global)
        code_ext = {c: i for i, c in enumerate(ext_names)}
    else:
        Mext, code_ext = Mtr, code_of_name
    te_code = np.array([code_ext[c] for c in te_chem], dtype=np.int64)
    st30.cache5_put(f"test_{ROLE_TAG}",
                    xattn_predict(net, fz_tr, te["meta"], te["C"],
                                  np.arange(len(te["meta"])), Mext, te_code, device))

    report["runtime_seconds"] = round(time.time() - t_start, 1)
    (RESULTS / "step6_xattn_training.json").write_text(json.dumps(report, indent=2))
    log(f"=== stage complete in {time.time() - t_start:.0f}s -> "
        f"results/step6_xattn_training.json ===")


if __name__ == "__main__":
    sys.exit(main())
