"""Step 4.3 -- multi-task PyTorch MLP-ResNet over the proteome.

Architecture rationale
----------------------
The GBDTs model one ``(sample, protein)`` **cell** at a time, which is why their
design matrix has 5.6 M rows. The network instead treats one **sample** as one
training example and emits the whole 5,243-protein vector at once. Three
consequences, all of them deliberate:

1. **The rubric's primary metric becomes directly differentiable.** Module 2 is
   75% ``pcc_per_sample_mean`` -- the Pearson correlation *across proteins within
   a sample*. That quantity does not exist inside a per-cell minibatch, so a
   per-cell model can only approach it through a squared-error surrogate. Here it
   is computed exactly, per example, and optimised as a loss term.

2. **Cross-protein structure is shared.** Proteins co-respond in modules; a
   vector-valued head can exploit that covariance, whereas a per-cell model sees
   each protein independently.

3. **The effective sample size becomes brutally visible.** There are only 5,920
   training samples, so the model is small and heavily regularised. This is the
   honest view of the problem: the 5.6 M cell count never was 5.6 M independent
   observations.

Generalisation to novel entities is handled by **entity dropout** rather than by
training four regime specialists. During training, the strain / batch / plate
embedding of a random subset of examples is replaced by the reserved
``__UNSEEN__`` token, so a single network learns to predict with those identities
missing -- exactly the state it will be served in on the novel-strain splits. The
chemical is represented *only* by its RDKit fingerprint and descriptors, never by
an identity embedding, which forces chemical generalisation through structure.

Honest early stopping
---------------------
Stopping on ``val_*`` would make every downstream number tuned-on. The schedule
is therefore selected on the inner-dev cohort carved out of ``train`` by
:func:`step4_common.load_inner_context`, and the model is then refit on all train
rows for the selected number of epochs. Both artefacts are kept: the inner-fit
model supplies the honest out-of-fold predictions the stacker needs, and the
refit model supplies the ``val``/``test`` predictions.

Outputs
-------
workflow/models_step4/dl_*.pt                 network weights
data/step4_cache/dl_delta_{inner,val}.npy     cached delta predictions
results/step4_dl_training.json                curves, config, timings
figures/step4_dl_training.png                 loss / metric curves
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

SESSION, DATA, RESULTS, FIGURES = S4.SESSION, S4.DATA, S4.RESULTS, S4.FIGURES
WORKFLOW, MODELS4, SEED = S4.WORKFLOW, S4.MODELS4, S4.SEED
CHEM_COL = S4.CHEM_COL
log = S4.log

#: Categorical identities the network embeds. The chemical is deliberately
#: absent: it is represented by structure only, so a novel compound is not a
#: novel token.
CAT_EMB = ["Strains", "data_source", "instrument", "Medium", "Yeast_cell_plate"]

#: Probability of replacing an identity embedding with __UNSEEN__ during
#: training, per field. Higher for the entities that actually go missing on the
#: OOD splits (strain) and for the high-cardinality batch fields.
ENTITY_DROPOUT = {
    "Strains": 0.35,
    "data_source": 0.20,
    "instrument": 0.20,
    "Medium": 0.20,
    "Yeast_cell_plate": 0.40,
}

UNSEEN = "__UNSEEN__"


# ---------------------------------------------------------------------------
# Feature assembly (sample level)
# ---------------------------------------------------------------------------
class SampleFeaturizer:
    """Builds sample-level network inputs, fitted on one cohort of rows.

    All normalisation statistics, categorical vocabularies and per-protein
    imputation values are estimated on ``fit_mask`` rows only, so serving a
    held-out cohort cannot leak information back into the fit.
    """

    def __init__(self, meta: pd.DataFrame, C: np.ndarray, Y: np.ndarray, fit_mask: np.ndarray):
        self.mol = S4.load_mol_features("full")
        self.desc_cols = [c for c in self.mol.columns if not c.startswith("fp_")]
        self.fp_cols = [c for c in self.mol.columns if c.startswith("fp_")]

        # --- molecular descriptor standardisation (over fit-set compounds) ---
        fit_chems = sorted(set(meta.loc[fit_mask, CHEM_COL].astype(str)))
        sub = self.mol.loc[[c for c in fit_chems if c in self.mol.index], self.desc_cols]
        self.desc_mu = sub.mean().to_numpy("float32")
        self.desc_sd = sub.std().replace(0.0, 1.0).to_numpy("float32")
        self.desc_sd = np.where(self.desc_sd > 1e-6, self.desc_sd, 1.0).astype("float32")

        # --- categorical vocabularies (fit rows only, plus UNSEEN) ----------
        self.vocab: dict[str, dict[str, int]] = {}
        for col in CAT_EMB:
            lv = sorted(set(meta.loc[fit_mask, col].astype(str)))
            self.vocab[col] = {v: i for i, v in enumerate(lv)}
            self.vocab[col][UNSEEN] = len(lv)

        # --- numeric covariates --------------------------------------------
        self.num_cols = ["pert_time_num", "temperature_num", "well_row", "well_col"]
        nm = meta.loc[fit_mask, self.num_cols].astype("float32").to_numpy()
        nm = np.where(np.isfinite(nm), nm, 0.0)
        nm[:, 0] = np.log1p(np.clip(nm[:, 0], 0, None))
        self.num_mu = nm.mean(0)
        self.num_sd = np.where(nm.std(0) > 1e-6, nm.std(0), 1.0)

        # --- control-profile imputation (per protein, fit rows) -------------
        with np.errstate(all="ignore"):
            cm = np.nanmean(C[fit_mask], axis=0)
            ym = np.nanmean(Y[fit_mask], axis=0)
        gm = float(np.nanmedian(Y[fit_mask]))
        self.c_impute = np.where(np.isfinite(cm), cm, np.where(np.isfinite(ym), ym, gm)).astype(
            "float32"
        )
        self.c_mu = float(np.nanmean(C[fit_mask])) if np.isfinite(C[fit_mask]).any() else gm
        self.c_sd = float(np.nanstd(C[fit_mask])) or 1.0
        self.p = C.shape[1]

    # -- dimensions -------------------------------------------------------
    @property
    def n_desc(self) -> int:
        return len(self.desc_cols)

    @property
    def n_fp(self) -> int:
        return len(self.fp_cols)

    @property
    def n_num(self) -> int:
        return len(self.num_cols) + 1  # + control detection fraction

    def cat_sizes(self) -> list[int]:
        return [len(self.vocab[c]) for c in CAT_EMB]

    # -- transform --------------------------------------------------------
    def transform(self, meta: pd.DataFrame, C: np.ndarray, idx: np.ndarray) -> dict:
        """Build the input tensors for the given sample rows."""
        m = meta.iloc[idx]
        chems = m[CHEM_COL].astype(str).to_numpy()
        loc = self.mol.index.get_indexer(pd.Index(chems))
        if (loc < 0).any():
            bad = sorted(set(chems[loc < 0].tolist()))[:8]
            raise KeyError(f"compounds without molecular features: {bad}")
        mol_np = self.mol.to_numpy("float32")
        d_idx = [self.mol.columns.get_loc(c) for c in self.desc_cols]
        f_idx = [self.mol.columns.get_loc(c) for c in self.fp_cols]
        desc = (mol_np[np.ix_(loc, d_idx)] - self.desc_mu) / self.desc_sd
        fp = mol_np[np.ix_(loc, f_idx)]

        cats = np.stack(
            [
                m[col]
                .astype(str)
                .map(lambda v, c=col: self.vocab[c].get(v, self.vocab[c][UNSEEN]))
                .to_numpy("int64")
                for col in CAT_EMB
            ],
            axis=1,
        )

        nm = m[self.num_cols].astype("float32").to_numpy()
        nm = np.where(np.isfinite(nm), nm, 0.0)
        nm[:, 0] = np.log1p(np.clip(nm[:, 0], 0, None))
        nm = (nm - self.num_mu) / self.num_sd

        Cr = C[idx]
        det = np.isfinite(Cr)
        ctrl = np.where(det, Cr, self.c_impute[None, :]).astype("float32")
        ctrl = (ctrl - self.c_mu) / self.c_sd
        num = np.concatenate([nm, det.mean(axis=1, keepdims=True)], axis=1).astype("float32")

        return {
            "desc": desc.astype("float32"),
            "fp": fp.astype("float32"),
            "cats": cats,
            "num": num,
            "ctrl": ctrl,
        }

    def state(self) -> dict:
        """Serialisable description of the fitted transform."""
        return {
            "desc_cols": self.desc_cols,
            "n_fp": self.n_fp,
            "cat_sizes": {c: len(self.vocab[c]) for c in CAT_EMB},
            "num_cols": self.num_cols,
        }


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def build_model(fz: SampleFeaturizer, width: int = 512, n_blocks: int = 3, emb_dim: int = 16,
                p_drop: float = 0.25):
    """Multi-task MLP-ResNet with two proteome-wide heads."""
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

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList(
                [nn.Embedding(n, emb_dim) for n in fz.cat_sizes()]
            )
            # The control profile is 5,243-dimensional; project it down before
            # it can dominate the representation.
            self.ctrl_proj = nn.Sequential(
                nn.Linear(fz.p, 256), nn.GELU(), nn.Dropout(p_drop)
            )
            self.fp_proj = nn.Sequential(
                nn.Linear(fz.n_fp, 128), nn.GELU(), nn.Dropout(p_drop)
            )
            d_in = 256 + 128 + fz.n_desc + fz.n_num + emb_dim * len(fz.cat_sizes())
            self.stem = nn.Sequential(nn.Linear(d_in, width), nn.GELU())
            self.blocks = nn.Sequential(*[ResBlock(width, p_drop) for _ in range(n_blocks)])
            self.norm = nn.LayerNorm(width)
            self.head_delta = nn.Linear(width, fz.p)
            self.head_abs = nn.Linear(width, fz.p)
            nn.init.zeros_(self.head_delta.weight)
            nn.init.zeros_(self.head_delta.bias)

        def forward(self, desc, fp, cats, num, ctrl):
            e = [emb(cats[:, i]) for i, emb in enumerate(self.embs)]
            h = torch.cat([self.ctrl_proj(ctrl), self.fp_proj(fp), desc, num] + e, dim=1)
            h = self.norm(self.blocks(self.stem(h)))
            return self.head_delta(h), self.head_abs(h)

    return Net()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def masked_mse(pred, target, mask):
    """Mean squared error over finite target cells."""
    import torch

    n = mask.sum()
    if n == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    d = (pred - target) * mask
    return (d * d).sum() / n


def neg_pearson_per_sample(pred, target, mask, min_n: int = 3):
    """1 - mean per-sample Pearson correlation across proteins.

    This is the differentiable form of Module 2's dominant submetric
    (``pcc_per_sample_mean``, 75% of a 25%-weighted module), computed within the
    same axis the harness uses. Samples with fewer than ``min_n`` observed
    proteins or zero variance are skipped, mirroring ``harness.masked_pcc``.
    """
    import torch

    n = mask.sum(dim=1)
    ok = n >= min_n
    if not bool(ok.any()):
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    m, x, y = mask[ok], pred[ok], target[ok]
    cnt = m.sum(dim=1, keepdim=True)
    xm = (x * m).sum(1, keepdim=True) / cnt
    ym = (y * m).sum(1, keepdim=True) / cnt
    xc, yc = (x - xm) * m, (y - ym) * m
    num = (xc * yc).sum(1)
    den = torch.sqrt((xc * xc).sum(1) * (yc * yc).sum(1) + 1e-8)
    r = num / den
    return 1.0 - r.mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_epochs(
    model,
    fz,
    meta,
    C,
    D,
    Y,
    fit_idx,
    dev_idx,
    n_epochs,
    device,
    lr=2e-3,
    wd=1e-2,
    batch=64,
    w_abs=0.3,
    w_pcc=1.0,
    seed=SEED,
    label="train",
    eval_every=1,
):
    """Train for ``n_epochs``; if ``dev_idx`` is given, track the dev objective.

    Returns
    -------
    history : list[dict]
    best : dict
        Best epoch by dev per-sample delta PCC (or last epoch if no dev set).
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, n_epochs * int(np.ceil(len(fit_idx) / batch)))
    )

    fit_feats = fz.transform(meta, C, fit_idx)
    Dt = torch.from_numpy(np.nan_to_num(D[fit_idx], nan=0.0))
    Dm = torch.from_numpy(np.isfinite(D[fit_idx]).astype("float32"))
    Yt = torch.from_numpy(np.nan_to_num(Y[fit_idx], nan=0.0))
    Ym = torch.from_numpy(np.isfinite(Y[fit_idx]).astype("float32"))
    T = {k: torch.from_numpy(v) for k, v in fit_feats.items()}

    dev_pack = None
    if dev_idx is not None and len(dev_idx):
        df = fz.transform(meta, C, dev_idx)
        dev_pack = (
            {k: torch.from_numpy(v).to(device) for k, v in df.items()},
            torch.from_numpy(np.nan_to_num(D[dev_idx], nan=0.0)).to(device),
            torch.from_numpy(np.isfinite(D[dev_idx]).astype("float32")).to(device),
        )

    # UNSEEN token id per embedded field, for entity dropout.
    unseen_ids = torch.tensor(
        [fz.vocab[c][UNSEEN] for c in CAT_EMB], dtype=torch.long
    )
    drop_p = torch.tensor([ENTITY_DROPOUT[c] for c in CAT_EMB], dtype=torch.float32)

    rng = np.random.default_rng(seed)
    history: list[dict] = []
    best = {"epoch": -1, "dev_pcc": -np.inf, "state": None}
    t0 = time.time()

    for ep in range(1, n_epochs + 1):
        model.train()
        order = rng.permutation(len(fit_idx))
        tot, nb = 0.0, 0
        for s in range(0, len(order), batch):
            sel = torch.from_numpy(order[s : s + batch])
            cats = T["cats"][sel].clone()
            # entity dropout -> __UNSEEN__
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
            loss = (
                masked_mse(pd_, dt, dm)
                + w_abs * masked_mse(pa_, yt, ym)
                + w_pcc * neg_pearson_per_sample(pd_, dt, dm)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1

        rec = {"epoch": ep, "train_loss": tot / max(1, nb)}
        if dev_pack is not None and (ep % eval_every == 0 or ep == n_epochs):
            model.eval()
            with torch.no_grad():
                f, dt, dm = dev_pack
                pd_, _ = model(f["desc"], f["fp"], f["cats"], f["num"], f["ctrl"])
                rec["dev_pcc"] = float(1.0 - neg_pearson_per_sample(pd_, dt, dm))
                rec["dev_mse"] = float(masked_mse(pd_, dt, dm))
            if rec["dev_pcc"] > best["dev_pcc"]:
                best = {
                    "epoch": ep,
                    "dev_pcc": rec["dev_pcc"],
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }
        history.append(rec)
        if ep % 5 == 0 or ep == 1 or ep == n_epochs:
            extra = (
                f" dev_pcc={rec.get('dev_pcc', float('nan')):.4f}" if "dev_pcc" in rec else ""
            )
            log(f"  [{label}] epoch {ep}/{n_epochs} loss={rec['train_loss']:.4f}{extra}"
                f" | {time.time() - t0:.0f}s")

    if best["state"] is None:
        best = {
            "epoch": n_epochs,
            "dev_pcc": float("nan"),
            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        }
    return history, best


def predict(model, fz, meta, C, idx, device, batch: int = 128) -> np.ndarray:
    """Predict the delta matrix for the given samples."""
    import torch

    model.eval()
    out = np.empty((len(idx), fz.p), dtype="float32")
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            sl = idx[s : s + batch]
            f = fz.transform(meta, C, sl)
            t = {k: torch.from_numpy(v).to(device) for k, v in f.items()}
            pd_, _ = model(t["desc"], t["fp"], t["cats"], t["num"], t["ctrl"])
            out[s : s + len(sl)] = pd_.cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 4

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODELS4.mkdir(parents=True, exist_ok=True)
    log("=== Step 4.3: multi-task MLP-ResNet ===")
    log(f"torch {torch.__version__} device={device} epochs={args.epochs}")

    ctx = S4.load_context()
    meta, Y, D, C = ctx["meta"], ctx["Y"], ctx["D"], ctx["C"]
    masks, VS = ctx["masks"], ctx["VS"]
    inner = S4.load_inner_context(ctx)

    # ---- stage 1: fit on inner_fit, select the epoch on inner_dev ---------
    fit_idx = np.flatnonzero(inner["fit_mask"])
    dev_idx = inner["dev_idx"]
    log(f"stage 1: inner_fit n={len(fit_idx)}  inner_dev n={len(dev_idx)}")

    fz_in = SampleFeaturizer(meta, C, Y, inner["fit_mask"])
    log(f"  featuriser: {fz_in.n_desc} descriptors, {fz_in.n_fp} fp bits, "
        f"cat sizes {fz_in.cat_sizes()}, ctrl dim {fz_in.p}")

    model = build_model(fz_in, width=args.width, n_blocks=args.blocks).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  parameters: {n_par:,}")

    hist_in, best_in = run_epochs(
        model, fz_in, meta, C, D, Y, fit_idx, dev_idx, args.epochs, device,
        lr=args.lr, label="inner",
    )
    log(f"  best inner-dev epoch = {best_in['epoch']} (dev per-sample PCC "
        f"{best_in['dev_pcc']:.4f})")

    model.load_state_dict(best_in["state"])
    torch.save(best_in["state"], MODELS4 / "dl_inner.pt")
    d_inner = predict(model, fz_in, meta, C, dev_idx, device)
    S4.cache_put("dl_delta_inner", d_inner)

    # ---- stage 2: refit on all train rows for the selected schedule -------
    train_mask = masks[VS.TRAIN_SPLIT]
    tr_idx = np.flatnonzero(train_mask)
    # Row count grows by 1/(1 - inner holdout); scale the epoch budget the same
    # way rather than early-stopping against data we are about to train on.
    scale = len(fit_idx) / max(1, len(tr_idx))
    n_ep2 = max(5, int(round(best_in["epoch"] * (1.0 / max(scale, 1e-6)) ** 0.0)))
    n_ep2 = max(5, best_in["epoch"])
    log(f"stage 2: refit on all {len(tr_idx)} train rows for {n_ep2} epochs")

    fz_tr = SampleFeaturizer(meta, C, Y, train_mask)
    model2 = build_model(fz_tr, width=args.width, n_blocks=args.blocks).to(device)
    hist_tr, best_tr = run_epochs(
        model2, fz_tr, meta, C, D, Y, tr_idx, None, n_ep2, device,
        lr=args.lr, label="refit",
    )
    torch.save(model2.state_dict(), MODELS4 / "dl_refit.pt")

    eval_idx = np.flatnonzero(masks["all_val"])
    d_val = predict(model2, fz_tr, meta, C, eval_idx, device)
    S4.cache_put("dl_delta_val", d_val)

    # ---- report ----------------------------------------------------------
    rep = {
        "step": "4c_deep_learning",
        "seed": SEED,
        "torch": torch.__version__,
        "device": device,
        "architecture": {
            "type": "multi-task MLP-ResNet, sample-level input -> proteome-wide output",
            "width": args.width,
            "n_blocks": args.blocks,
            "n_parameters": int(n_par),
            "heads": ["delta (fold-change)", "abs (log2 abundance)"],
            "control_profile_dim": int(fz_in.p),
            "entity_dropout": ENTITY_DROPOUT,
            "chemical_representation": (
                "RDKit fingerprint + descriptors only; no chemical identity embedding, so a "
                "novel compound is not a novel token"
            ),
        },
        "loss": (
            "masked MSE(delta) + 0.3 * masked MSE(abundance) + 1.0 * (1 - per-sample Pearson "
            "of delta across proteins); the Pearson term is the differentiable form of Module "
            "2's dominant submetric"
        ),
        "epoch_selection": {
            "protocol": (
                "epoch chosen on the inner-dev cohort carved from train; val_* is never used "
                "for stopping"
            ),
            "best_inner_epoch": int(best_in["epoch"]),
            "best_inner_dev_pcc": float(best_in["dev_pcc"]),
            "refit_epochs": int(n_ep2),
        },
        "cohorts": {
            "inner_fit": int(len(fit_idx)),
            "inner_dev": int(len(dev_idx)),
            "train": int(len(tr_idx)),
            "all_val": int(len(eval_idx)),
        },
        "featuriser": fz_in.state(),
        "history_inner": hist_in,
        "history_refit": hist_tr,
        "smoke": bool(args.smoke),
    }
    S4.write_json(RESULTS / "step4_dl_training.json", rep)

    # ---- curves ----------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ep = [h["epoch"] for h in hist_in]
    axes[0].plot(ep, [h["train_loss"] for h in hist_in], label="inner_fit loss", lw=1.2)
    axes[0].plot(
        [h["epoch"] for h in hist_tr], [h["train_loss"] for h in hist_tr],
        label="refit loss", lw=1.2, ls="--",
    )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss")
    axes[0].set_title("Optimisation")
    axes[0].legend(frameon=False, fontsize=7)

    dv = [(h["epoch"], h["dev_pcc"]) for h in hist_in if "dev_pcc" in h]
    if dv:
        axes[1].plot([a for a, _ in dv], [b for _, b in dv], color="#C4442E", lw=1.4)
        axes[1].axvline(best_in["epoch"], color="k", ls=":", lw=0.9,
                        label=f"selected epoch {best_in['epoch']}")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("inner-dev per-sample PCC (delta)")
        axes[1].set_title("Honest epoch selection (inner-dev, never val)")
        axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Step 4.3 -- multi-task MLP-ResNet training", fontsize=9.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"step4_dl_training.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {FIGURES / 'step4_dl_training.png'}")
    log("=== deep learning stage complete ===")


if __name__ == "__main__":
    main()
