"""Step 5.4 -- graph-attention ResNet and protein-cluster non-negative stacking.

Two things happen here, and only the second one is expected to move the score.

1. Graph-attention ResNet (``--stage gnn``)
-------------------------------------------
The Step-4 deep member ended in a dense ``Linear(512, 5243)`` head: 5,243
independent output rows, no notion that two co-regulated proteins should respond
alike. This model replaces that head with a **factorised, graph-attentive** one::

    delta_hat = (Z W_p) @ (E + Attn(E))^T + b

where ``E`` is a learned per-protein embedding **initialised from the STRING
spectral coordinates**, and ``Attn`` is single-head additive attention over each
protein's top-``k`` STRING neighbours. Two proteins that interact therefore share
representational capacity by construction rather than by coincidence, and the
head holds ~0.4M parameters instead of 5.4M.

On top of that the loss carries a graph-smoothness penalty
``lambda_graph * tr(P^T L P) / (n p)`` with ``L`` the normalised Laplacian
(cross-domain prior XD3). A ``lambda_graph = 0`` control is run on one fold so the
penalty's contribution is *measured*, not asserted.

The two-term data loss (masked MSE and per-sample correlation) is balanced by the
**gradient-cosine rule** taken from *ProteinTalks* (focused prior FD4): clip the
cosine of the two task gradients at 1.0, form ``0.01 x cosine``, and shift weight
toward the correlation task only when the gradients conflict.

The network is cross-fitted over the same 5 LCGO folds as the GBDTs, so its
out-of-fold matrix is honest, then refitted on all train rows for the val and test
cohorts.

2. Protein-cluster non-negative stacking (``--stage stack``)
------------------------------------------------------------
This is the change that is expected to matter, and the reason is the rubric's own
algebra (cross-domain prior XD5). Module 2 is built entirely from Pearson
correlations of Delta and so is invariant to positive rescaling; Module 1 contains
two R^2 terms and is strongly scale-sensitive. The shortfall of a non-negative
weight vector from 1 is therefore a free shrinkage toward the control anchor: it
buys Module 1 at no cost to Module 2. Step 4 exploited that with one scalar per
(regime, role).

But R^2 is dominated by per-protein dynamic range. A hyper-abundant enzyme with a
wide response distribution tolerates a large predicted Delta; a low-abundance
regulatory factor, whose measured variance is mostly noise, does not. The optimal
shrinkage is thus **protein-dependent**, and a single scalar per regime is forced
to average over that. So the weight tensor becomes

    W in R^{regimes x roles x protein-clusters}

with the clusters from Step 5.2 (STRING spectral topology + train-only abundance
and response statistics). :mod:`step5_clusterscore` evaluates this exactly, and
is validated against the real harness on adversarial cluster-weight tensors before
the optimiser is allowed to run.

What is fitted where -- the part that keeps the number honest
-------------------------------------------------------------
* weights are fitted **only** on the train-OOF cohort (all 5,078 rows, every
  member prediction out-of-fold);
* they are then **frozen** and applied to ``val_*`` exactly once;
* the reported headline is that frozen-weight val total.

The val-tuned optimum is also computed and reported, explicitly labelled
``OPTIMISTIC``, purely to quantify how much headroom the weight space contains.

The ablation ladder separates the two changes that Step 5 makes at once -- a
bigger calibration cohort and a bigger weight space -- so the credit is
attributable instead of asserted:

======================================  ================================
rung                                    isolates
======================================  ================================
Step-4 scalar weights, inner cohort     the published baseline
scalar weights, OOF cohort              value of cross-fitting alone
cluster weights, OOF cohort             value of the cluster extension
======================================  ================================

Outputs
-------
results/step5_model_scores.json          every candidate, per module, both cohorts
results/step5_cluster_weights.json       the frozen weight tensor actually used
results/step5_bootstrap_ci.json          paired bootstrap CIs on the margins
figures/step5_cluster_stacking_weights.png
figures/step5_performance_comparison.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_common as S4  # noqa: E402

DATA, RESULTS, FIGURES, WORKFLOW = S4.DATA, S4.RESULTS, S4.FIGURES, S4.WORKFLOW
MODELS4 = S4.MODELS4
SEED, CHEM_COL, log = S4.SEED, S4.CHEM_COL, S4.log

CACHE5 = DATA / "step5_cache"
MODELS5 = WORKFLOW / "models_step5"
REGIMES = ("full", "chem_novel", "strain_novel", "both_novel")
BENCH_TOTAL = 0.445442994
STEP4_VAL_TOTAL = 0.477376776  # the honest Step-4 headline this step must beat

#: role -> (oof cache name, val cache name, test cache name).
#: Every role must exist on all three cohorts, produced at matched
#: hyper-parameters with only the fit set differing -- otherwise the frozen
#: weights are calibrated against a member that is not the one they are applied to.
ROLES5: dict[str, tuple[str, str, str]] = {
    "gbdt_tab": ("oof_lgb_tab", "val_lgb_tab", "test_lgb_tab"),
    "gbdt_mol": ("oof_lgb_mol", "val_lgb_mol", "test_lgb_mol"),
    "gbdt_mol3d": ("oof_lgb_mol3d", "val_lgb_mol3d", "test_lgb_mol3d"),
    "bench": ("oof_bench", "val_bench", "test_bench"),
    "gnn": ("oof_gnn", "val_gnn", "test_gnn"),
    "dl": ("oof_dl", "val_dl", "test_dl"),
}

CLUSTER_COL_DEFAULT = "k8"


def load_module(path: Path, name: str):
    """Import a module whose filename starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cache5_get(name: str) -> np.ndarray | None:
    p = CACHE5 / f"{name}.npy"
    return np.load(p) if p.exists() else None


def cache5_put(name: str, arr: np.ndarray) -> None:
    CACHE5.mkdir(parents=True, exist_ok=True)
    np.save(CACHE5 / f"{name}.npy", np.ascontiguousarray(arr, dtype="float32"))
    log(f"  cached {name} {arr.shape}")


def load_graph() -> dict:
    """Load the STRING adjacency, spectral embedding and protein clusters."""
    import scipy.sparse as sp

    z = np.load(DATA / "step5_protein_graph.npz", allow_pickle=False)
    adj = sp.csr_matrix(
        (z["adj_data"], z["adj_indices"], z["adj_indptr"]), shape=tuple(z["adj_shape"])
    )
    clus = pd.read_parquet(DATA / "step5_protein_clusters.parquet")
    return {
        "adj": adj,
        "embedding": z["embedding"],
        "proteins": list(z["proteins"]),
        "graph_source": str(z["graph_source"][0]),
        "clusters": clus,
    }


# ===========================================================================
# Stage 1: graph-attention ResNet
# ===========================================================================
def topk_neighbours(adj, k: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` neighbours per protein by interaction score.

    The full graph averages ~93 neighbours per protein; attending over all of them
    would dominate the step cost for little gain, since STRING scores are strongly
    ranked. Proteins with fewer than ``k`` neighbours are padded with themselves
    and the pad is masked out, so an isolated protein simply keeps its own
    embedding rather than receiving an arbitrary neighbour.
    """
    n = adj.shape[0]
    idx = np.tile(np.arange(n, dtype=np.int64)[:, None], (1, k))
    msk = np.zeros((n, k), dtype="float32")
    for i in range(n):
        s, e = adj.indptr[i], adj.indptr[i + 1]
        if e <= s:
            continue
        nb, w = adj.indices[s:e], adj.data[s:e]
        take = np.argsort(-w)[:k]
        m = len(take)
        idx[i, :m] = nb[take]
        msk[i, :m] = 1.0
    log(f"  attention neighbourhood: k={k}, "
        f"{float(msk.sum(axis=1).mean()):.1f} real neighbours per protein on average, "
        f"{int((msk.sum(axis=1) == 0).sum())} proteins with none")
    return idx, msk


def build_gnn(fz, emb0: np.ndarray, nbr_idx, nbr_msk, width: int = 512,
              n_blocks: int = 3, emb_dim: int = 16, d_prot: int = 64,
              p_drop: float = 0.25, n_heads: int = 4):
    """Graph-attention ResNet with a factorised, STRING-initialised output head."""
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, d, p):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d)
            )

        def forward(self, x):
            return x + self.net(x)

    class ProteinGraphAttention(nn.Module):
        """Multi-head additive attention over each protein's STRING neighbours."""

        def __init__(self, d, heads):
            super().__init__()
            self.h = heads
            self.dh = d // heads
            self.q = nn.Linear(d, d, bias=False)
            self.k = nn.Linear(d, d, bias=False)
            self.v = nn.Linear(d, d, bias=False)
            self.o = nn.Linear(d, d)
            self.norm = nn.LayerNorm(d)
            # Zero-initialised output projection: the model starts as the
            # no-attention baseline and has to earn any graph mixing it uses.
            nn.init.zeros_(self.o.weight)
            nn.init.zeros_(self.o.bias)

        def forward(self, E, idx, msk):
            p, d = E.shape
            q = self.q(E).reshape(p, self.h, self.dh)
            kk = self.k(E)[idx].reshape(p, -1, self.h, self.dh)
            vv = self.v(E)[idx].reshape(p, -1, self.h, self.dh)
            att = (q[:, None] * kk).sum(-1) / (self.dh ** 0.5)   # (p, k, h)
            att = att.masked_fill(msk[:, :, None] < 0.5, float("-inf"))
            att = torch.softmax(att, dim=1)
            att = torch.nan_to_num(att, nan=0.0)  # proteins with no neighbour
            out = (att[..., None] * vv).sum(1).reshape(p, d)
            return self.norm(E + self.o(out))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(n, emb_dim) for n in fz.cat_sizes()])
            self.ctrl_proj = nn.Sequential(nn.Linear(fz.p, 256), nn.GELU(), nn.Dropout(p_drop))
            self.fp_proj = nn.Sequential(nn.Linear(fz.n_fp, 128), nn.GELU(), nn.Dropout(p_drop))
            d_in = 256 + 128 + fz.n_desc + fz.n_num + emb_dim * len(fz.cat_sizes())
            self.stem = nn.Sequential(nn.Linear(d_in, width), nn.GELU())
            self.blocks = nn.Sequential(*[ResBlock(width, p_drop) for _ in range(n_blocks)])
            self.norm = nn.LayerNorm(width)

            # Protein embeddings, initialised from the STRING spectral coordinates
            # (scaled to unit RMS so the initial scale is comparable to a default
            # nn.Linear init) and then free to move.
            e0 = np.asarray(emb0[:, :d_prot], dtype="float32")
            if e0.shape[1] < d_prot:
                e0 = np.pad(e0, ((0, 0), (0, d_prot - e0.shape[1])))
            rms = float(np.sqrt((e0 ** 2).mean())) or 1.0
            self.E = nn.Parameter(torch.from_numpy(e0 / rms * 0.5))
            self.register_buffer("nbr_idx", torch.from_numpy(nbr_idx))
            self.register_buffer("nbr_msk", torch.from_numpy(nbr_msk))
            self.gat = ProteinGraphAttention(d_prot, n_heads)

            self.proj_delta = nn.Linear(width, d_prot)
            self.proj_abs = nn.Linear(width, d_prot)
            self.bias_delta = nn.Parameter(torch.zeros(fz.p))
            self.bias_abs = nn.Parameter(torch.zeros(fz.p))
            # Zero-initialised delta projection reproduces Step 4's convention:
            # the initial fold-change prediction is exactly the control anchor.
            nn.init.zeros_(self.proj_delta.weight)
            nn.init.zeros_(self.proj_delta.bias)

        def protein_repr(self):
            return self.gat(self.E, self.nbr_idx, self.nbr_msk)

        def forward(self, desc, fp, cats, num, ctrl):
            e = [emb(cats[:, i]) for i, emb in enumerate(self.embs)]
            h = torch.cat([self.ctrl_proj(ctrl), self.fp_proj(fp), desc, num] + e, dim=1)
            h = self.norm(self.blocks(self.stem(h)))
            P = self.protein_repr()
            return (self.proj_delta(h) @ P.T + self.bias_delta,
                    self.proj_abs(h) @ P.T + self.bias_abs)

    return Net()


def train_gnn(model, dl, fz, meta, C, D, Y, fit_idx, dev_idx, n_epochs, device,
              L_sparse, lambda_graph: float, lr: float = 2e-3, wd: float = 1e-2,
              batch: int = 64, w_abs: float = 0.3, seed: int = SEED,
              label: str = "gnn", eval_every: int = 5) -> tuple[list, dict]:
    """Train with cosine-annealed warm restarts, gradient-cosine task balancing
    and a Laplacian smoothness penalty.

    The two data-loss terms are balanced by the *ProteinTalks* gradient-cosine
    rule rather than a fixed ratio: the cosine of the two task gradients (measured
    on the shared trunk's last layer, which is where they actually compete) is
    clipped at 1.0, an adjustment of ``0.01 x cosine`` is formed, and weight is
    moved toward the correlation task only when the cosine is negative, i.e. only
    when the tasks genuinely conflict.
    """
    import torch

    CAT_EMB, ENTITY_DROPOUT, UNSEEN = dl.CAT_EMB, dl.ENTITY_DROPOUT, dl.UNSEEN
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)

    feats = fz.transform(meta, C, fit_idx)
    T = {k: torch.from_numpy(v) for k, v in feats.items()}
    Dt = torch.from_numpy(np.nan_to_num(D[fit_idx], nan=0.0))
    Dm = torch.from_numpy(np.isfinite(D[fit_idx]).astype("float32"))
    Yt = torch.from_numpy(np.nan_to_num(Y[fit_idx], nan=0.0))
    Ym = torch.from_numpy(np.isfinite(Y[fit_idx]).astype("float32"))

    dev_pack = None
    if dev_idx is not None and len(dev_idx):
        df = fz.transform(meta, C, dev_idx)
        dev_pack = (
            {k: torch.from_numpy(v).to(device) for k, v in df.items()},
            torch.from_numpy(np.nan_to_num(D[dev_idx], nan=0.0)).to(device),
            torch.from_numpy(np.isfinite(D[dev_idx]).astype("float32")).to(device),
        )

    unseen_ids = torch.tensor([fz.vocab[c][UNSEEN] for c in CAT_EMB], dtype=torch.long)
    drop_p = torch.tensor([ENTITY_DROPOUT[c] for c in CAT_EMB], dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, int(np.ceil(len(fit_idx) / batch)))
    # Cosine annealing with warm restarts: the objective is a mixture of a
    # squared-error term and a correlation term whose optima differ in scale, so
    # a schedule that periodically re-heats escapes the basin the MSE term pulls
    # toward early on. T_0 is one fifth of the run, doubling each restart.
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, (n_epochs * steps_per_epoch) // 7), T_mult=2, eta_min=lr * 0.02
    )

    Lc = L_sparse.tocoo() if lambda_graph > 0 else None
    if Lc is not None:
        Li = torch.from_numpy(np.vstack([Lc.row, Lc.col]).astype("int64"))
        Lv = torch.from_numpy(Lc.data.astype("float32"))
        Lt = torch.sparse_coo_tensor(Li, Lv, Lc.shape).coalesce().to(device)

    rng = np.random.default_rng(seed)
    hist: list[dict] = []
    best = {"epoch": -1, "dev_pcc": -np.inf, "state": None}
    w_pcc, w_mse = 1.0, 1.0
    t0 = time.time()

    for ep in range(1, n_epochs + 1):
        model.train()
        order = rng.permutation(len(fit_idx))
        tot = n_b = 0.0
        cos_acc, n_cos = 0.0, 0
        for s in range(0, len(order), batch):
            sel = torch.from_numpy(order[s : s + batch])
            cats = T["cats"][sel].clone()
            m = torch.rand(cats.shape) < drop_p[None, :]
            cats = torch.where(m, unseen_ids[None, :].expand_as(cats), cats)

            desc = T["desc"][sel].to(device)
            fp = T["fp"][sel].to(device)
            num = T["num"][sel].to(device)
            ctrl = T["ctrl"][sel].to(device)
            cats = cats.to(device)
            dt, dm = Dt[sel].to(device), Dm[sel].to(device)
            yt, ym = Yt[sel].to(device), Ym[sel].to(device)

            pd_, pa_ = model(desc, fp, cats, num, ctrl)
            l_mse = dl.masked_mse(pd_, dt, dm)
            l_pcc = dl.neg_pearson_per_sample(pd_, dt, dm)
            l_abs = dl.masked_mse(pa_, yt, ym)

            # ---- gradient-cosine task balancing (prior FD4) ---------------
            # Measured on the shared trunk's last linear layer: that is the
            # parameter both heads pull on, so it is where a conflict is real.
            if s == 0:
                shared = model.blocks[-1].net[-1].weight
                g1 = torch.autograd.grad(l_mse, shared, retain_graph=True,
                                         allow_unused=True)[0]
                g2 = torch.autograd.grad(l_pcc, shared, retain_graph=True,
                                         allow_unused=True)[0]
                if g1 is not None and g2 is not None:
                    denom = g1.norm() * g2.norm() + 1e-12
                    cos = float(torch.clamp((g1 * g2).sum() / denom, max=1.0))
                    cos_acc += cos
                    n_cos += 1
                    if cos < 0.0:  # gradients conflict -> favour the priority task
                        adj = 0.01 * abs(cos)
                        w_pcc = float(min(2.0, w_pcc + adj))
                        w_mse = float(max(0.2, w_mse - adj))

            loss = w_mse * l_mse + w_abs * l_abs + w_pcc * l_pcc
            if lambda_graph > 0:
                # tr(P^T L P) / (n p): mean graph-Dirichlet energy of the
                # predicted fold-change matrix.
                LP = torch.sparse.mm(Lt, pd_.T)
                loss = loss + lambda_graph * (pd_.T * LP).sum() / pd_.numel()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            n_b += 1

        rec = {"epoch": ep, "train_loss": tot / max(1, n_b), "w_mse": w_mse,
               "w_pcc": w_pcc, "lr": float(opt.param_groups[0]["lr"])}
        if n_cos:
            rec["grad_cosine"] = cos_acc / n_cos
        if dev_pack is not None and (ep % eval_every == 0 or ep == n_epochs):
            model.eval()
            with torch.no_grad():
                f, dt, dm = dev_pack
                pd_, _ = model(f["desc"], f["fp"], f["cats"], f["num"], f["ctrl"])
                rec["dev_pcc"] = float(1.0 - dl.neg_pearson_per_sample(pd_, dt, dm))
                rec["dev_mse"] = float(dl.masked_mse(pd_, dt, dm))
            if rec["dev_pcc"] > best["dev_pcc"]:
                best = {"epoch": ep, "dev_pcc": rec["dev_pcc"],
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
        hist.append(rec)
        if ep == 1 or ep % 5 == 0 or ep == n_epochs:
            extra = (f" dev_pcc={rec['dev_pcc']:.4f}" if "dev_pcc" in rec else "")
            log(f"  [{label}] epoch {ep}/{n_epochs} loss={rec['train_loss']:.4f}"
                f"{extra} w_mse={w_mse:.3f} w_pcc={w_pcc:.3f} | {time.time() - t0:.0f}s")

    if best["state"] is None:
        best = {"epoch": n_epochs, "dev_pcc": float("nan"),
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    return hist, best


def gnn_predict(model, fz, meta, C, idx, device, batch: int = 128) -> np.ndarray:
    """Delta-head predictions for ``idx`` rows of ``(meta, C)``."""
    import torch

    model = model.to(device).eval()
    out = np.empty((len(idx), fz.p), dtype="float32")
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            sl = idx[s : s + batch]
            f = fz.transform(meta, C, sl)
            t = {k: torch.from_numpy(v).to(device) for k, v in f.items()}
            pd_, _ = model(t["desc"], t["fp"], t["cats"], t["num"], t["ctrl"])
            out[s : s + len(sl)] = pd_.cpu().numpy()
    return out


def stage_gnn(args) -> None:
    """Cross-fit the GNN over the LCGO folds, then refit on all train rows."""
    import scipy.sparse as sp
    import torch

    dl = load_module(WORKFLOW / "24_train_deep_learning.py", "dl24")
    lc = load_module(WORKFLOW / "29_lcgo_oof_matrix.py", "lcgo29")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__} device={device} epochs={args.epochs}")
    MODELS5.mkdir(parents=True, exist_ok=True)

    G = load_graph()
    adj = G["adj"]
    deg = np.asarray(adj.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0))
    Dm_ = sp.diags(dinv)
    L = (sp.eye(adj.shape[0], format="csr") - Dm_ @ adj @ Dm_).astype("float32")
    log(f"  graph: {G['graph_source']}, {adj.nnz // 2} undirected edges")
    nbr_idx, nbr_msk = topk_neighbours(adj, k=args.attn_k)

    ctx = S4.load_context()
    meta, Y, D, C = ctx["meta"], ctx["Y"], ctx["D"], ctx["C"]
    masks, VS = ctx["masks"], ctx["VS"]
    train_mask = masks[VS.TRAIN_SPLIT]
    folds = lc.build_folds(meta, train_mask, seed=SEED)
    tr_idx = np.flatnonzero(train_mask)
    pos_of_row = -np.ones(len(meta), dtype=np.int64)
    pos_of_row[tr_idx] = np.arange(len(tr_idx))

    # ``--arch mlp`` cross-fits the Step-4 dense-head architecture instead, so a
    # deep member calibrated on the LCGO folds exists either way. That matters
    # because Step 4's stacking gave its deep member large weight, and dropping the
    # deep role entirely would confound the Step-5 comparison with a member-pool
    # change rather than isolating the cluster extension.
    tag = "gnn" if args.arch == "gnn" else "dl"
    report: dict = {"step": "5_4_gnn", "seed": SEED, "device": device,
                    "architecture": args.arch, "role_tag": tag,
                    "graph_source": G["graph_source"], "attn_k": args.attn_k,
                    "lambda_graph": args.lambda_graph if args.arch == "gnn" else 0.0,
                    "epochs": args.epochs, "folds": {}}
    lam = args.lambda_graph if args.arch == "gnn" else 0.0

    def make_net(fz):
        if args.arch == "gnn":
            return build_gnn(fz, G["embedding"], nbr_idx, nbr_msk, width=args.width,
                             n_blocks=args.blocks, d_prot=args.d_prot)
        return dl.build_model(fz, width=args.width, n_blocks=args.blocks)

    oof = np.full((len(tr_idx), Y.shape[1]), np.nan, dtype="float32")
    n_folds = 1 if args.smoke else lc.N_FOLDS
    for f in range(n_folds):
        fit_mask = folds["fits"][f]
        dev_rows = np.flatnonzero(folds["fold"] == f)
        log(f"\n===== GNN fold {f}: fit n={int(fit_mask.sum())} dev n={len(dev_rows)} =====")
        fz = dl.SampleFeaturizer(meta, C, Y, fit_mask)
        net = make_net(fz)
        n_par = sum(p.numel() for p in net.parameters())
        log(f"  {n_par:,} parameters")
        hist, best = train_gnn(
            net, dl, fz, meta, C, D, Y, np.flatnonzero(fit_mask), dev_rows,
            args.epochs, device, L, lam, lr=args.lr,
            label=f"{tag}_fold{f}", eval_every=args.eval_every,
        )
        net.load_state_dict(best["state"])
        oof[pos_of_row[dev_rows]] = gnn_predict(net, fz, meta, C, dev_rows, device)
        torch.save(best["state"], MODELS5 / f"{tag}_fold{f}.pt")
        report["folds"][f"fold{f}"] = {
            "n_fit": int(fit_mask.sum()), "n_dev": int(len(dev_rows)),
            "n_parameters": int(n_par), "best_epoch": best["epoch"],
            "best_dev_pcc": best["dev_pcc"], "history": hist,
        }
        log(f"  fold {f}: best epoch {best['epoch']} dev per-sample PCC "
            f"{best['dev_pcc']:.4f}")
        del net, fz

        # Pre-specified control on fold 0 only: does the Laplacian penalty help?
        if f == 0 and lam > 0 and not args.skip_control:
            log("  running the lambda_graph = 0 control on fold 0 ...")
            fz0 = dl.SampleFeaturizer(meta, C, Y, fit_mask)
            net0 = make_net(fz0)
            _, best0 = train_gnn(
                net0, dl, fz0, meta, C, D, Y, np.flatnonzero(fit_mask), dev_rows,
                args.epochs, device, L, 0.0, lr=args.lr, label="gnn_fold0_lam0",
                eval_every=args.eval_every,
            )
            report["lambda_graph_control_fold0"] = {
                "lambda_graph": 0.0,
                "best_dev_pcc": best0["dev_pcc"],
                "best_dev_pcc_with_penalty": best["dev_pcc"],
                "delta_from_penalty": best["dev_pcc"] - best0["dev_pcc"],
                "note": (
                    "single-fold comparison of the graph-smoothness penalty against its own "
                    "control; one fold is a noisy estimate and the difference is reported as a "
                    "diagnostic, not as an established effect"
                ),
            }
            log(f"  lambda_graph control: dev PCC {best0['dev_pcc']:.4f} (lam=0) vs "
                f"{best['dev_pcc']:.4f} (lam={args.lambda_graph}) -> "
                f"{best['dev_pcc'] - best0['dev_pcc']:+.4f}")
            del net0, fz0

    if args.smoke:
        # Architecture / plumbing check only: one fold, few epochs, nothing cached
        # and no report written, so a smoke run can never be mistaken for a real one.
        log("\nSMOKE RUN: one fold, no artefact written. Verifying the prediction block ...")
        got = oof[pos_of_row[np.flatnonzero(folds["fold"] == 0)]]
        log(f"  fold-0 predictions {got.shape}: finite={bool(np.isfinite(got).all())} "
            f"mean={float(got.mean()):+.4f} sd={float(got.std()):.4f} "
            f"range [{float(got.min()):+.3f}, {float(got.max()):+.3f}]")
        if not np.isfinite(got).all():
            raise AssertionError("the GNN produced non-finite predictions")
        log("=== smoke run OK (no artefact written) ===")
        return

    n_complete = int(np.isfinite(oof).all(axis=1).sum())
    log(f"\n{tag} OOF matrix: {n_complete}/{len(tr_idx)} rows complete")
    cache5_put(f"oof_{tag}", np.nan_to_num(oof, nan=0.0))
    report["oof_rows_complete"] = n_complete

    # ---- refit on all train rows for the val and test cohorts ------------
    log(f"\n===== {tag} refit on all train rows =====")
    n_ep2 = max(5, int(np.median([report["folds"][k]["best_epoch"]
                                  for k in report["folds"]])))
    log(f"  epoch count taken from the fold-wise selections (median): {n_ep2}")
    fz_tr = dl.SampleFeaturizer(meta, C, Y, train_mask)
    net = make_net(fz_tr)
    hist2, best2 = train_gnn(
        net, dl, fz_tr, meta, C, D, Y, tr_idx, None, n_ep2, device, L,
        lam, lr=args.lr, label=f"{tag}_refit", eval_every=args.eval_every,
    )
    net.load_state_dict(best2["state"])
    torch.save(best2["state"], MODELS5 / f"{tag}_refit.pt")
    report["refit"] = {"n_epochs": n_ep2, "history": hist2}

    val_idx = np.flatnonzero(masks["all_val"])
    cache5_put(f"val_{tag}", gnn_predict(net, fz_tr, meta, C, val_idx, device))

    log(f"predicting the test cohort with the {tag} model ...")
    te = ctx["S3"].load_test(ctx["proteins"], ctx["meta_all"], ctx["M_all"])
    cache5_put(f"test_{tag}", gnn_predict(net, fz_tr, te["meta"], te["C"],
                                          np.arange(len(te["meta"])), device))

    S4.write_json(RESULTS / f"step5_{tag}_training.json", report)
    log("=== stage gnn complete ===")


# ===========================================================================
# Stage 2: cluster stacking
# ===========================================================================
def load_oof_cohort(ctx: dict, roles: list[str]) -> dict:
    """Assemble the train-OOF calibration cohort."""
    meta, Y, D = ctx["meta"], ctx["Y"], ctx["D"]
    z = np.load(CACHE5 / "oof_baselines.npz", allow_pickle=False)
    tr_idx = z["train_idx"]
    regimes = np.asarray(z["regime"], dtype=object)

    members = {}
    for role in roles:
        arr = cache5_get(ROLES5[role][0])
        if arr is None:
            log(f"  !! role '{role}' missing on the OOF cohort "
                f"({ROLES5[role][0]}.npy absent); dropped")
            continue
        members[role] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    meta_eval = meta.iloc[tr_idx].copy().reset_index(drop=True)
    meta_eval["split_final"] = [S4.REGIME_TO_SPLIT[r] for r in regimes]
    log(f"  OOF cohort: {len(tr_idx)} samples, roles {sorted(members)}, "
        f"regimes {dict(pd.Series(regimes).value_counts())}")
    return {
        "idx": tr_idx, "meta_eval": meta_eval,
        "Y": Y[tr_idx], "D": D[tr_idx], "C_h": ctx["C_harness"][tr_idx],
        "y_fb": z["y_fallback"], "mu_ctx": z["mu_ctx"], "mu_drug": z["mu_drug"],
        "members": members, "regimes": regimes, "roles": sorted(members),
    }


def load_val_cohort(ctx: dict, roles: list[str]) -> dict:
    """Assemble the held-out validation cohort with the matched members."""
    meta, Y, D = ctx["meta"], ctx["Y"], ctx["D"]
    mask = ctx["masks"]["all_val"]
    idx = np.flatnonzero(mask)

    members = {}
    for role in roles:
        if role == "bench":
            # The analytic member is a projection of train-fitted group means, not
            # a trained model, so it is derived here rather than cached: caching it
            # would only create a way for it to fall out of step with the
            # train-frozen tables it comes from.
            members[role] = np.nan_to_num(
                (ctx["Y_bench"][mask] - ctx["C_harness"][mask]).astype("float32"),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            continue
        arr = cache5_get(ROLES5[role][1])
        if arr is None:
            log(f"  !! role '{role}' missing on val ({ROLES5[role][1]}.npy absent); dropped")
            continue
        members[role] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    y_fb = S4.cache_get("val_y_fallback")
    if y_fb is None:
        y_fb = ctx["y_fallback"][mask]
    rp = S4.cache_path("val_regimes")
    regimes = (np.load(rp, allow_pickle=False).astype(object) if rp.exists()
               else S4.regimes_for_samples(ctx["enc"], meta, idx))
    log(f"  val cohort: {len(idx)} samples, roles {sorted(members)}, "
        f"regimes {dict(pd.Series(regimes).value_counts())}")
    return {
        "idx": idx, "meta_eval": meta.loc[mask].reset_index(drop=True),
        "Y": Y[idx], "D": D[idx], "C_h": ctx["C_harness"][idx],
        "y_fb": y_fb, "mu_ctx": ctx["mu_ctx"][mask], "mu_drug": ctx["mu_drug"][mask],
        "members": members, "regimes": regimes, "roles": sorted(members),
    }


def score_cluster_blend(coh: dict, W: np.ndarray, roles: list[str],
                        cluster_of_col: np.ndarray) -> dict:
    """Exact harness score for a cluster-weight tensor on one cohort."""
    import step5_clusterscore as CS

    d = CS.blend_clusters(coh, W, roles, REGIMES, cluster_of_col)
    yh, dh = S4.reconstruct(d, coh["C_h"], coh["y_fb"])
    return S4.score(coh["Y"], yh, coh["D"], dh, coh["meta_eval"],
                    coh["mu_ctx"], coh["mu_drug"])


def optimise_clusters(fast, m4: float, W0: np.ndarray, n_sweeps: int = 12,
                      time_budget: float = 900.0, alpha: float = 0.0,
                      label: str = "opt") -> tuple[np.ndarray, float, dict]:
    """Coordinate ascent over the non-negative cluster-weight tensor.

    Warm-started from the scalar optimum broadcast across clusters, which is both
    faster and better conditioned than a cold start: the cluster extension then
    only has to find the *departures* from a shared weight, and it can always fall
    back to the scalar solution if a cluster carries no signal.

    ``alpha`` optionally penalises the spread of a role's weights across clusters,
    ``alpha * sum_{r,k} var_c(W[r,k,:])``. This is a shrinkage toward the scalar
    solution and is the knob that trades cluster resolution against
    weight-estimation variance; ``alpha = 0`` reproduces unpenalised coordinate
    ascent.
    """
    nR, nK, nC = W0.shape
    W = W0.copy()
    n_eval = 0
    t0 = time.time()
    truncated = False

    def obj(Wc):
        nonlocal n_eval
        n_eval += 1
        v = fast.total(Wc, m4=m4)
        if alpha > 0.0 and nC > 1:
            v -= alpha * float(np.sum(np.var(Wc, axis=2)))
        return v

    s = obj(W)
    log(f"  [{label}] start total={s:.6f} (alpha={alpha})")
    grids = [
        np.array([-0.30, -0.20, -0.12, -0.06, -0.03, 0.0, 0.03, 0.06, 0.12, 0.20, 0.30]),
        np.array([-0.05, -0.025, -0.012, 0.0, 0.012, 0.025, 0.05]),
        np.array([-0.008, -0.004, -0.002, 0.0, 0.002, 0.004, 0.008]),
    ]
    for gi, grid in enumerate(grids):
        improved = True
        sweep = 0
        while improved and sweep < n_sweeps:
            improved = False
            sweep += 1
            for ri in range(nR):
                for ki in range(nK):
                    for ci in range(nC):
                        base = W[ri, ki, ci]
                        for v in np.clip(base + grid, 0.0, None):
                            if v == base:
                                continue
                            W[ri, ki, ci] = v
                            sv = obj(W)
                            if sv > s + 1e-10:
                                s, base, improved = sv, v, True
                            else:
                                W[ri, ki, ci] = base
                if time.time() - t0 > time_budget:
                    truncated = True
                    break
            log(f"  [{label}] grid {gi} sweep {sweep}: total={s:.6f} "
                f"({n_eval} evals, {time.time() - t0:.0f}s)")
            if truncated:
                break
        if truncated:
            break
    if truncated:
        log(f"  [{label}] !! SEARCH TRUNCATED at the {time_budget:.0f}s budget after "
            f"{n_eval} evaluations; the reported weights are the best found so far")
    log(f"  [{label}] best={s:.6f} after {n_eval} evaluations in {time.time() - t0:.0f}s")
    return W, s, {"n_evaluations": n_eval, "seconds": round(time.time() - t0, 1),
                  "truncated": truncated, "alpha": alpha}


def per_sample_components(coh: dict, W: np.ndarray, roles: list[str],
                          cluster_of_col: np.ndarray) -> dict:
    """Per-sample submetric vectors, for the paired bootstrap.

    Only the submetrics that are literally means over samples are returned. Those
    carry 0.5875 of the total module weight; the pooled and per-protein-mean terms
    are not sample means, so resampling samples does not give them a valid
    bootstrap distribution and they are excluded rather than approximated.
    """
    import harness as H
    import step5_clusterscore as CS

    d = CS.blend_clusters(coh, W, roles, REGIMES, cluster_of_col)
    D = coh["D"]
    mu_c, mu_d = coh["mu_ctx"], coh["mu_drug"]
    split = coh["meta_eval"]["split_final"].to_numpy()

    out = {
        "m2_pcc_per_sample": np.asarray(H.masked_pcc(D, d, axis=1), dtype="float64"),
        "resid_ctx_pcc_per_sample": np.asarray(
            H.masked_pcc(D - mu_c, d - mu_c, axis=1), dtype="float64"),
        "resid_drug_pcc_per_sample": np.asarray(
            H.masked_pcc(D - mu_d, d - mu_d, axis=1), dtype="float64"),
        "split": split,
    }
    return out


def bootstrap_margin(comp_a: dict, comp_b: dict, n_boot: int = 1000,
                     seed: int = SEED) -> dict:
    """Paired non-parametric bootstrap over samples for the per-sample component.

    Both candidates are evaluated on the *same* resampled sample sets, so the
    difference is paired and the CI reflects only the disagreement between them,
    not the shared sampling noise of the cohort.
    """
    rng = np.random.default_rng(seed)
    split = comp_a["split"]
    n = len(split)
    idx_by_split = {s: np.flatnonzero(split == s) for s in np.unique(split)}

    # weight of each per-sample-mean submetric in the total
    terms = [
        ("m2_pcc_per_sample", None, 0.25 * 0.75),
        ("resid_ctx_pcc_per_sample", "val_chem_only", 0.20 * 0.75),
        ("resid_drug_pcc_per_sample", "val_strain_only", 0.20 * 0.75),
        ("m2_pcc_per_sample", "val_both", 0.05 / 3.0),
        ("resid_ctx_pcc_per_sample", "val_both", 0.05 / 3.0),
        ("resid_drug_pcc_per_sample", "val_both", 0.05 / 3.0),
        ("m2_pcc_per_sample", "val_time", 0.05 / 3.0),
        ("resid_ctx_pcc_per_sample", "val_time", 0.05 / 3.0),
        ("resid_drug_pcc_per_sample", "val_time", 0.05 / 3.0),
    ]
    total_weight = sum(w for _, _, w in terms)

    def component(comp, boot: np.ndarray | None) -> float:
        acc = 0.0
        for key, sp, w in terms:
            v = comp[key]
            rows = np.arange(n) if sp is None else idx_by_split.get(sp, np.array([], int))
            if boot is not None:
                # resample within the relevant row set, preserving its size
                if not len(rows):
                    continue
                rows = rows[boot[: len(rows)] % len(rows)]
            vv = v[rows]
            vv = vv[np.isfinite(vv)]
            acc += w * float(min(max(vv.mean(), 0.0), 1.0)) if vv.size else 0.0
        return acc

    pt_a, pt_b = component(comp_a, None), component(comp_b, None)
    diffs = np.empty(n_boot)
    a_s = np.empty(n_boot)
    b_s = np.empty(n_boot)
    for b in range(n_boot):
        boot = rng.integers(0, n, size=n)
        a_s[b] = component(comp_a, boot)
        b_s[b] = component(comp_b, boot)
        diffs[b] = a_s[b] - b_s[b]
        if (b + 1) % 200 == 0:
            log(f"    bootstrap {b + 1}/{n_boot}")
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_boot": n_boot,
        "component_weight_of_total": round(total_weight, 6),
        "point_a": pt_a, "point_b": pt_b, "point_diff": pt_a - pt_b,
        "diff_mean": float(diffs.mean()), "diff_sd": float(diffs.std(ddof=1)),
        "diff_ci95": [float(lo), float(hi)],
        "frac_bootstrap_replicates_favouring_a": float((diffs > 0).mean()),
        "a_ci95": [float(x) for x in np.percentile(a_s, [2.5, 97.5])],
        "b_ci95": [float(x) for x in np.percentile(b_s, [2.5, 97.5])],
        "scope_note": (
            "the bootstrap covers only the submetrics that are literally means over samples "
            f"({total_weight:.4f} of the 1.0 module weight). Pooled and per-protein-mean "
            "submetrics are not sample means, so resampling samples does not yield a valid "
            "bootstrap distribution for them and they are excluded rather than approximated. "
            "The CI is therefore a CI on that component of the margin, not on the full total."
        ),
    }


def weights_table(W: np.ndarray, roles: list[str]) -> dict:
    """Human-readable nested dict of the cluster-weight tensor."""
    return {
        regime: {
            role: [round(float(x), 4) for x in W[ri, ki]]
            for ki, role in enumerate(roles)
        }
        for ri, regime in enumerate(REGIMES)
    }


def stage_stack(args) -> None:
    """Fit the cluster weights on the OOF cohort, freeze, evaluate once on val."""
    import step5_clusterscore as CS

    np.random.seed(SEED)
    FIGURES.mkdir(exist_ok=True)
    ctx = S4.load_context()

    G = load_graph()
    clus = G["clusters"]
    if args.cluster_col not in clus.columns:
        raise SystemExit(f"cluster column {args.cluster_col!r} absent; "
                         f"available: {[c for c in clus.columns if c != 'protein']}")

    wanted = [r for r in args.roles.split(",") if r]
    log("assembling cohorts ...")
    coh_oof = load_oof_cohort(ctx, wanted)
    coh_val = load_val_cohort(ctx, wanted)
    roles = [r for r in wanted if r in coh_oof["roles"] and r in coh_val["roles"]]
    if not roles:
        raise SystemExit("no member role is available on both cohorts")
    log(f"roles usable on both cohorts: {roles}")
    dropped = [r for r in wanted if r not in roles]
    if dropped:
        log(f"!! roles dropped for lack of a matched member on one cohort: {dropped}")

    results: dict[str, dict] = {}

    # ---- individual members and the control anchor, both cohorts ---------
    ones = np.ones(len(G["proteins"]), dtype=np.int64) * 0
    for name, coh in (("oof", coh_oof), ("val", coh_val)):
        for ki, role in enumerate(roles):
            W = np.zeros((len(REGIMES), len(roles), 1))
            W[:, ki, 0] = 1.0
            r = score_cluster_blend(coh, W, roles, ones)
            results[f"{name}:{role}"] = r
            log(f"  {name:4s} {role:11s} total={r['total_score']:.6f}")
        r0 = score_cluster_blend(coh, np.zeros((len(REGIMES), len(roles), 1)), roles, ones)
        results[f"{name}:control_anchor"] = r0
        log(f"  {name:4s} {'zero':11s} total={r0['total_score']:.6f}")

    m4_oof = float(results["oof:" + roles[0]]["module_scores"]["m4_dep"])
    m4_val = float(results["val:" + roles[0]]["module_scores"]["m4_dep"])

    # ---- rung 1: scalar weights on the OOF cohort (nC = 1) ---------------
    log("\n=== rung 1: SCALAR weights fitted on the OOF cohort ===")
    t0 = time.time()
    fast_s = CS.ClusterFastScorer(coh_oof, roles, REGIMES, ones, 1)
    log(f"  scorer built in {time.time() - t0:.1f}s")
    val_s = fast_s.validate(
        lambda W: score_cluster_blend(coh_oof, W, roles, ones), n_trials=5, seed=SEED
    )
    W0 = np.full((len(REGIMES), len(roles), 1), 1.0 / max(len(roles), 1))
    W_scal, s_scal, tr_scal = optimise_clusters(
        fast_s, m4_oof, W0, time_budget=args.budget, label="scalar/oof"
    )
    r_oof_scal = score_cluster_blend(coh_oof, W_scal, roles, ones)
    results["oof:scalar_frozen"] = r_oof_scal
    r_val_scal = score_cluster_blend(coh_val, W_scal, roles, ones)
    results["val:scalar_oof_frozen"] = r_val_scal
    log(f"*** rung 1: val total with scalar OOF-frozen weights = "
        f"{r_val_scal['total_score']:.6f} ***")

    # ---- rung 2: cluster weights, warm-started from the scalar optimum ----
    ladder: dict[str, dict] = {}
    best_cluster = {"col": None, "n_clusters": 1, "W": W_scal, "oof": r_oof_scal,
                    "alpha": 0.0}
    cluster_cols = [c for c in args.k_ladder.split(",") if c]
    for col in cluster_cols:
        if col not in clus.columns:
            log(f"  cluster column {col!r} absent; skipped")
            continue
        cl = clus[col].to_numpy(np.int64)
        nC = int(cl.max()) + 1
        if nC == 1:
            ladder[col] = {"n_clusters": 1,
                           "oof_total": float(r_oof_scal["total_score"]),
                           "note": "identical to the scalar rung by construction"}
            continue
        log(f"\n=== cluster ladder: {col} ({nC} clusters, sizes "
            f"{np.bincount(cl).tolist()}) ===")
        t0 = time.time()
        fast_c = CS.ClusterFastScorer(coh_oof, roles, REGIMES, cl, nC)
        log(f"  scorer built in {time.time() - t0:.1f}s")
        vc = fast_c.validate(
            lambda W, cl=cl: score_cluster_blend(coh_oof, W, roles, cl),
            n_trials=5, seed=SEED
        )
        Wstart = np.repeat(W_scal, nC, axis=2)
        entry: dict = {"n_clusters": nC, "cluster_sizes": np.bincount(cl).tolist(),
                       "fast_validation": {k: v for k, v in vc.items() if k != "rows"},
                       "by_alpha": {}}
        for alpha in [float(a) for a in args.alphas.split(",") if a]:
            Wc, sc, trc = optimise_clusters(
                fast_c, m4_oof, Wstart, time_budget=args.budget, alpha=alpha,
                label=f"{col}/a{alpha}",
            )
            r_oof = score_cluster_blend(coh_oof, Wc, roles, cl)
            entry["by_alpha"][str(alpha)] = {
                "oof_total_exact": float(r_oof["total_score"]),
                "fast_objective_at_optimum": float(sc),
                "optimiser": trc,
                "weight_spread_across_clusters": float(np.mean(np.std(Wc, axis=2))),
            }
            log(f"  {col} alpha={alpha}: OOF exact total {r_oof['total_score']:.6f}")
            # Selection is on the OOF cohort only -- val is not consulted.
            if r_oof["total_score"] > best_cluster["oof"]["total_score"]:
                best_cluster = {"col": col, "n_clusters": nC, "W": Wc, "oof": r_oof,
                                "alpha": alpha, "cl": cl}
        ladder[col] = entry

    # ---- freeze and evaluate once on val --------------------------------
    W_fin = best_cluster["W"]
    cl_fin = best_cluster.get("cl", ones)
    log(f"\nselected on the OOF cohort: cluster column "
        f"{best_cluster['col'] or 'scalar (nC=1)'} with "
        f"{best_cluster['n_clusters']} clusters, alpha={best_cluster['alpha']}")
    r_val_fin = score_cluster_blend(coh_val, W_fin, roles, cl_fin)
    results["val:cluster_oof_frozen"] = r_val_fin
    results["oof:cluster_frozen"] = best_cluster["oof"]
    headline = float(r_val_fin["total_score"])
    log(f"*** rung 2: val total with CLUSTER OOF-frozen weights = {headline:.6f} "
        f"(benchmark {BENCH_TOTAL:.6f}, Step 4 {STEP4_VAL_TOTAL:.6f}) ***")

    # ---- diagnostic only: val-tuned optimum (OPTIMISTIC) -----------------
    log("\ndiagnostic: fitting weights directly on val to measure headroom "
        "(OPTIMISTIC, never reported as the result) ...")
    fast_v = CS.ClusterFastScorer(coh_val, roles, REGIMES, cl_fin,
                                  best_cluster["n_clusters"])
    _ = fast_v.validate(
        lambda W: score_cluster_blend(coh_val, W, roles, cl_fin), n_trials=3,
        seed=SEED + 1,
    )
    W_vt, _, _ = optimise_clusters(
        fast_v, m4_val, W_fin.copy(), time_budget=args.budget / 2, label="val_tuned",
    )
    r_val_tuned = score_cluster_blend(coh_val, W_vt, roles, cl_fin)
    results["val:cluster_val_tuned_OPTIMISTIC"] = r_val_tuned

    # ---- paired bootstrap on the per-sample component --------------------
    log("\npaired bootstrap: cluster-frozen vs scalar-frozen on val ...")
    comp_cluster = per_sample_components(coh_val, W_fin, roles, cl_fin)
    comp_scalar = per_sample_components(coh_val, W_scal, roles, ones)
    boot_vs_scalar = bootstrap_margin(comp_cluster, comp_scalar, n_boot=args.n_boot)
    log(f"  cluster - scalar per-sample component: "
        f"{boot_vs_scalar['point_diff']:+.6f} "
        f"95% CI [{boot_vs_scalar['diff_ci95'][0]:+.6f}, "
        f"{boot_vs_scalar['diff_ci95'][1]:+.6f}]")

    log("paired bootstrap: cluster-frozen vs the benchmark predictor on val ...")
    W_bench = np.zeros((len(REGIMES), len(roles), 1))
    if "bench" in roles:
        W_bench[:, roles.index("bench"), 0] = 1.0
    comp_bench = per_sample_components(coh_val, W_bench, roles, ones)
    boot_vs_bench = bootstrap_margin(comp_cluster, comp_bench, n_boot=args.n_boot)
    log(f"  cluster - benchmark-member per-sample component: "
        f"{boot_vs_bench['point_diff']:+.6f} "
        f"95% CI [{boot_vs_bench['diff_ci95'][0]:+.6f}, "
        f"{boot_vs_bench['diff_ci95'][1]:+.6f}]")

    S4.write_json(
        RESULTS / "step5_bootstrap_ci.json",
        {
            "step": "5_4_bootstrap",
            "seed": SEED,
            "cluster_frozen_vs_scalar_frozen": boot_vs_scalar,
            "cluster_frozen_vs_benchmark_member": boot_vs_bench,
            "method": (
                "paired non-parametric bootstrap over validation samples: both candidates are "
                "scored on the same resampled sample sets, so the shared cohort sampling noise "
                "cancels and the CI reflects only their disagreement"
            ),
        },
    )

    # ---- FD1: chem_novel performance stratified by chemical support -------
    strat = chem_support_stratification(ctx, coh_val, W_fin, roles, cl_fin)

    out = {
        "step": "5_4_cluster_stacking",
        "seed": SEED,
        "benchmark_total": BENCH_TOTAL,
        "step4_val_total": STEP4_VAL_TOTAL,
        "roles": roles,
        "roles_dropped": dropped,
        "role_sources": {r: {"oof": ROLES5[r][0], "val": ROLES5[r][1],
                             "test": ROLES5[r][2]} for r in roles},
        "graph_source": G["graph_source"],
        "selected": {
            "cluster_column": best_cluster["col"],
            "n_clusters": int(best_cluster["n_clusters"]),
            "alpha": best_cluster["alpha"],
            "selected_on": "the train-OOF cohort only; val was not consulted",
        },
        "protocol": {
            "weight_space": (
                "non-negative, per (availability regime x member role x protein cluster), NOT "
                "constrained to sum to one: the shortfall from one is a shrinkage toward the "
                "control anchor, which buys the scale-sensitive R^2 terms of Module 1 without "
                "costing the scale-invariant Pearson terms of Module 2. Making that shrinkage "
                "per-cluster is the point of Step 5, because R^2 is dominated by per-protein "
                "dynamic range"
            ),
            "fitted_on": (
                "the 5-fold LCGO out-of-fold cohort: all 5,078 train rows, every member "
                "prediction produced by a model that never saw the row, with encoders and both "
                "residual baselines refitted per fold"
            ),
            "evaluated_on": "val_* once, with the OOF-fitted weights frozen",
            "cluster_index": (
                "K-means on STRING spectral coordinates plus train-only abundance and response "
                "statistics, rank-transformed; fitted on train rows only so the index transfers "
                "to the test set the same way the regime routing does"
            ),
            "optimiser": (
                "coordinate ascent on a coarse-to-fine grid, warm-started from the scalar "
                "optimum broadcast across clusters, non-negativity by construction"
            ),
            "objective_implementation": (
                "modules 1-3 (0.95 of the weight) are evaluated by the cluster co-moment "
                "factorisation in step5_clusterscore, algebraically identical to the harness; "
                "module 4 is not a polynomial in the weights so it is held constant during the "
                "search and every reported total comes from the real harness"
            ),
            "fast_objective_validation_scalar": {k: v for k, v in val_s.items()
                                                 if k != "rows"},
        },
        "ablation_ladder": {
            "rung0_step4_scalar_inner_cohort": STEP4_VAL_TOTAL,
            "rung1_scalar_oof_cohort": float(r_val_scal["total_score"]),
            "rung2_cluster_oof_cohort": headline,
            "gain_from_cross_fitting_alone": float(
                r_val_scal["total_score"] - STEP4_VAL_TOTAL),
            "gain_from_the_cluster_extension": float(
                headline - r_val_scal["total_score"]),
            "note": (
                "rung 0 is the published Step-4 number, which used a different (weaker) member "
                "set as well as a smaller calibration cohort, so the first gain is not purely "
                "attributable to cross-fitting; rung 1 -> rung 2 is a clean A/B, since only the "
                "weight space changes"
            ),
        },
        "cluster_ladder": ladder,
        "frozen_weights": weights_table(W_fin, roles),
        "frozen_weights_scalar_rung": weights_table(W_scal, roles),
        "oof_total_at_frozen_weights": float(best_cluster["oof"]["total_score"]),
        "val_total_at_frozen_weights": headline,
        "val_total_val_tuned_OPTIMISTIC": float(r_val_tuned["total_score"]),
        "generalisation_gap_oof_to_val": float(
            headline - best_cluster["oof"]["total_score"]),
        "beats_benchmark": bool(headline > BENCH_TOTAL),
        "margin_vs_benchmark": float(headline - BENCH_TOTAL),
        "beats_step4": bool(headline > STEP4_VAL_TOTAL),
        "margin_vs_step4": float(headline - STEP4_VAL_TOTAL),
        "target_0_4850_met": bool(headline > 0.4850),
        "bootstrap": {
            "cluster_vs_scalar": {k: v for k, v in boot_vs_scalar.items()},
            "cluster_vs_benchmark_member": {k: v for k, v in boot_vs_bench.items()},
        },
        "chemical_support_stratification_FD1": strat,
        "module_scores": {k: v["module_scores"] for k, v in results.items()},
        "module_weights": next(iter(results.values()))["module_weights"],
        "totals": {k: float(v["total_score"]) for k, v in results.items()},
        "caveats": [
            "the LCGO fit sets hold ~3,000 rows against 5,078 for the members the frozen "
            "weights are applied to, so a member-strength mismatch remains; it is roughly half "
            "the Step-4 gap but it is not zero",
            "the val-tuned total quantifies headroom in the weight space and must not be read "
            "as an achieved score",
            "the bootstrap CI covers the per-sample-mean submetrics only; its scope is stated "
            "on the interval itself",
        ],
    }
    S4.write_json(RESULTS / "step5_cluster_weights.json", out)
    S4.write_json(RESULTS / "step5_model_scores.json", out)
    np.save(CACHE5 / "frozen_W.npy", W_fin)
    np.save(CACHE5 / "frozen_clusters.npy", cl_fin)
    (CACHE5 / "frozen_roles.json").write_text(json.dumps(roles), encoding="utf-8")

    make_weight_figure(W_fin, roles, cl_fin, clus, G)
    make_performance_figure(out)

    print("\n=== TOTALS ===")
    for k, v in sorted(out["totals"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:42s} {v:.6f}")
    print(f"\n  benchmark to beat        : {BENCH_TOTAL:.6f}")
    print(f"  Step-4 honest headline   : {STEP4_VAL_TOTAL:.6f}")
    print("  Step-5 target            : 0.485000")
    print(f"  HEADLINE (cluster weights frozen on OOF, applied to val): {headline:.6f} "
          f"({headline - BENCH_TOTAL:+.6f} vs benchmark, "
          f"{headline - STEP4_VAL_TOTAL:+.6f} vs Step 4)")
    print(f"  target 0.4850 met        : {out['target_0_4850_met']}")
    log("=== stage stack complete ===")


def chem_support_stratification(ctx, coh_val, W, roles, cl) -> dict:
    """Operationalise prior FD1: chem-novel score vs chemical support.

    A held-out compound with no structurally similar training compound has no
    support for a structure-activity extrapolation, and averaging it together with
    a well-supported one hides where the model actually fails.
    """
    import harness as H
    import step5_clusterscore as CS

    p = RESULTS / "step5_chemical_support.csv"
    if not p.exists():
        return {"skipped": "results/step5_chemical_support.csv absent "
                           "(run 27_knowledge_extraction.py)"}
    sup = pd.read_csv(p).set_index("compound")["max_tanimoto_to_train"].to_dict()

    d = CS.blend_clusters(coh_val, W, roles, REGIMES, cl)
    r = np.asarray(H.masked_pcc(coh_val["D"] - coh_val["mu_ctx"],
                                d - coh_val["mu_ctx"], axis=1), dtype="float64")
    meta = coh_val["meta_eval"]
    is_chem = meta["split_final"].to_numpy() == "val_chem_only"
    chem = meta[CHEM_COL].astype(str).to_numpy()
    tan = np.array([sup.get(c, np.nan) for c in chem])

    rows = []
    for lo, hi, lbl in ((0.0, 0.2, "0.0-0.2"), (0.2, 0.3, "0.2-0.3"),
                        (0.3, 0.5, "0.3-0.5"), (0.5, 1.01, "0.5-1.0")):
        m = is_chem & np.isfinite(tan) & (tan >= lo) & (tan < hi)
        v = r[m]
        v = v[np.isfinite(v)]
        rows.append({
            "max_tanimoto_to_train_bin": lbl,
            "n_val_chem_only_samples": int(m.sum()),
            "n_compounds": int(len(set(chem[m].tolist()))),
            "mean_residual_ctx_pcc_per_sample": float(v.mean()) if v.size else None,
        })
    log("  FD1 stratification of val_chem_only by chemical support:")
    for rr in rows:
        log(f"    Tanimoto {rr['max_tanimoto_to_train_bin']}: "
            f"n={rr['n_val_chem_only_samples']:5d} "
            f"({rr['n_compounds']} compounds) resid PCC="
            f"{rr['mean_residual_ctx_pcc_per_sample']}")
    return {
        "metric": "per-sample context-residual PCC on val_chem_only, the Module-3 S1 submetric",
        "bins": rows,
        "threshold_basis": (
            "the ~0.3 ECFP4 Tanimoto mark separating 'structurally dissimilar' is a virtual-"
            "screening convention, not a published threshold for this task"
        ),
    }


def make_weight_figure(W, roles, cl, clus, G) -> None:
    """Heatmaps of the fitted weight tensor plus the cluster response profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.linewidth": 0.6})
    nR, nK, nC = W.shape
    stat = pd.read_parquet(DATA / "step5_protein_stats.parquet")

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.6))
    vmax = float(max(W.max(), 1e-6))
    for ri, regime in enumerate(REGIMES):
        ax = axes.ravel()[ri]
        im = ax.imshow(W[ri], aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_yticks(range(nK))
        ax.set_yticklabels(roles, fontsize=7)
        ax.set_xticks(range(nC))
        ax.set_xticklabels([str(c) for c in range(nC)], fontsize=6)
        ax.set_xlabel("protein cluster (ordered by abundance)")
        ax.set_title(f"{regime}   (sum per cluster: "
                     f"{W[ri].sum(axis=0).min():.2f}-{W[ri].sum(axis=0).max():.2f})",
                     fontsize=8)
        for ki in range(nK):
            for ci in range(nC):
                ax.text(ci, ki, f"{W[ri, ki, ci]:.2f}", ha="center", va="center",
                        fontsize=5,
                        color="white" if W[ri, ki, ci] < 0.6 * vmax else "black")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, label="weight")

    ax = axes[1, 1]
    tot = W.sum(axis=1)  # (regime, cluster) total weight = 1 - shrinkage
    for ri, regime in enumerate(REGIMES):
        ax.plot(range(nC), tot[ri], "o-", lw=1.2, ms=4, label=regime)
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("protein cluster (ordered by abundance)")
    ax.set_ylabel("total weight (1.0 = no shrinkage)")
    ax.set_title("Shrinkage toward the control anchor varies by cluster")
    ax.legend(fontsize=6, frameon=False)

    ax = axes[1, 2]
    ab = [stat.loc[cl == c, "abundance_mean"].to_numpy() for c in range(nC)]
    bp = ax.boxplot(ab, showfliers=False, patch_artist=True, widths=0.6)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(plt.get_cmap("tab10")(i % 10))
        b.set_alpha(0.6)
        b.set_linewidth(0.5)
    for m in bp["medians"]:
        m.set_color("black")
        m.set_linewidth(0.9)
    ax.set_xlabel("protein cluster")
    ax.set_ylabel("train mean log2 abundance")
    ax.set_title("What the clusters are")

    fig.suptitle(
        "Step 5.4  Protein-cluster non-negative stacking weights "
        f"({nR} regimes x {nK} roles x {nC} clusters), fitted on the LCGO out-of-fold cohort",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"step5_cluster_stacking_weights.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {FIGURES / 'step5_cluster_stacking_weights.png'}")


def make_performance_figure(out: dict) -> None:
    """Ablation ladder, per-module comparison and the cluster-K sensitivity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))

    ax = axes[0]
    lad = out["ablation_ladder"]
    names = ["official\nbenchmark", "Step 4\nscalar / inner",
             "Step 5\nscalar / OOF", "Step 5\ncluster / OOF"]
    vals = [out["benchmark_total"], lad["rung0_step4_scalar_inner_cohort"],
            lad["rung1_scalar_oof_cohort"], lad["rung2_cluster_oof_cohort"]]
    cols = ["#9E9E9E", "#8FAADC", "#4472A8", "#2E7D32"]
    b = ax.bar(range(len(vals)), vals, color=cols, width=0.68)
    for i, (r, v) in enumerate(zip(b, vals)):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center",
                fontsize=7)
    ax.axhline(0.4850, color="#C62828", ls="--", lw=0.9)
    ax.text(len(vals) - 0.4, 0.4850, " target 0.4850", color="#C62828", fontsize=6,
            va="bottom", ha="right")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("harness total score")
    ax.set_ylim(0, max(max(vals), 0.4850) * 1.16)
    ax.set_title("Ablation ladder (all held-out val_*)")

    ax = axes[1]
    ms = out["module_scores"]
    mw = out["module_weights"]
    mods = list(mw)
    x = np.arange(len(mods))
    for off, (key, lbl, c) in enumerate((
        ("val:bench", "benchmark member", "#9E9E9E"),
        ("val:scalar_oof_frozen", "scalar / OOF", "#4472A8"),
        ("val:cluster_oof_frozen", "cluster / OOF", "#2E7D32"),
    )):
        if key not in ms:
            continue
        ax.bar(x + (off - 1) * 0.27, [ms[key].get(m, 0.0) for m in mods], width=0.26,
               label=lbl, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("m3_", "m3\n").replace("m1_", "m1\n")
                        .replace("m2_", "m2\n").replace("m4_", "m4\n") for m in mods],
                       fontsize=6)
    ax.set_ylabel("module score")
    ax.set_title("Per-module composition on val_*")
    ax.legend(fontsize=6, frameon=False)

    ax = axes[2]
    lad2 = out["cluster_ladder"]
    ks, tots = [], []
    for col, e in lad2.items():
        if "by_alpha" in e:
            best = max(v["oof_total_exact"] for v in e["by_alpha"].values())
        else:
            best = e.get("oof_total", np.nan)
        ks.append(e["n_clusters"])
        tots.append(best)
    if ks:
        o = np.argsort(ks)
        ax.plot(np.array(ks)[o], np.array(tots)[o], "o-", color="#2E7D32", lw=1.3, ms=5)
        sel = out["selected"]["n_clusters"]
        ax.axvline(sel, color="#C62828", ls="--", lw=0.9)
        ax.text(sel, min(tots), f" selected K={sel}", color="#C62828", fontsize=6,
                rotation=90, va="bottom")
    else:
        ax.text(0.5, 0.5, "no cluster rung ran", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="grey")
    ax.set_xlabel("number of protein clusters")
    ax.set_ylabel("OOF-cohort harness total")
    ax.set_title("Cluster-count sensitivity\n(selected on the OOF cohort, not on val)")

    fig.suptitle("Step 5  Protein-cluster stacking: what each change buys", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"step5_performance_comparison.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {FIGURES / 'step5_performance_comparison.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gnn", "stack"], default="stack")
    ap.add_argument("--roles", default="gbdt_tab,gbdt_mol,gbdt_mol3d,bench,gnn,dl")
    ap.add_argument("--arch", choices=["gnn", "mlp"], default="gnn",
                    help="gnn = graph-attention head; mlp = the Step-4 dense head")
    ap.add_argument("--cluster-col", default=CLUSTER_COL_DEFAULT)
    ap.add_argument("--k-ladder", default="k4,k8,k12,k16")
    ap.add_argument("--alphas", default="0.0,0.02")
    # Coordinate ascent at ~4 ms per evaluation converges in 60-150 s for the
    # tensor sizes here; this is a cap that rarely binds, sized so the whole
    # ladder (4 cluster columns x 2 alphas) cannot run away.
    ap.add_argument("--budget", type=float, default=450.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    # GNN stage
    ap.add_argument("--epochs", type=int, default=110)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--d-prot", type=int, default=64)
    ap.add_argument("--attn-k", type=int, default=16)
    ap.add_argument("--lambda-graph", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="one fold, no artefact written: plumbing check only")
    args = ap.parse_args()

    np.random.seed(SEED)
    log(f"=== Step 5.4: GNN and protein-cluster stacking (stage={args.stage}) ===")
    if args.stage == "gnn":
        stage_gnn(args)
    else:
        stage_stack(args)


if __name__ == "__main__":
    main()
