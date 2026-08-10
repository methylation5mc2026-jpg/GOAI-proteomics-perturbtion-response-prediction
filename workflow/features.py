"""Tabular feature engineering for the GBDT baselines (Step 3).

Design
------
The prediction target is a ``(n_samples, n_proteins)`` matrix, but a GBDT wants
rows.  Every ``(sample, protein)`` cell therefore becomes one row of a long
design matrix.  A cell's features come from three places:

**Sample-level** -- the experiment description: strain, medium, temperature,
``pert_time`` (numeric, so the tree can interpolate on the time grid), chemical,
plus the technical context ``data_source`` / ``instrument`` / plate / well.
Step 2 showed PC1 (29.8% of variance) is essentially ``data_source`` and that
conditioning on it is worth ~0.45 residual PCC on the 20%-weighted S1 module, so
the batch covariates are first-class features here, not nuisance terms.

**Protein-level** -- six statistics per protein (mean/sd of log2 abundance, mean
/ sd / mean-absolute ``Delta``, detection rate) computed on fit rows only.  These
carry protein identity in a low-dimensional, generalising form.  A raw 5,243-level
protein categorical is deliberately *not* used: the cell-level group-mean
features below are already protein-resolved (they are ``(group, protein)``
tables), so the identity information is present without a high-cardinality
categorical that mostly invites overfitting.

**Cell-level** -- the matched control abundance ``C[i, j]`` (available at
prediction time by construction: the competition defines
``Delta_pred = y_hat - y_control``, and each split ships its own controls), plus
a family of *group-mean ``Delta`` tables*.  Each table is a ``(n_groups,
n_proteins)`` array of the fit-set mean ``Delta`` for one grouping key.  Twelve
keys at different granularities are supplied rather than one hierarchical
fallback, so the model can perform the fallback itself:

============================  ==================================================
Table                         Available on ...
============================  ==================================================
``d_ctx``, ``d_ctxb``         seen strain  (S1, Time)
``d_strain``                  seen strain
``d_drug``, ``d_drug_time``   seen chemical (S2, Time)
``d_drug_strain``             seen strain *and* chemical (Time)
``d_batch``, ``d_batch_time``
``d_plate``, ``d_instr``      always (technical keys are shared across splits)
``d_time``, ``d_mtt``         always
============================  ==================================================

A group whose cell has fewer than :data:`MIN_CELL_N` finite fit observations is
NaN, and NaN is passed through to the model (LightGBM / XGBoost / CatBoost all
route missing values natively).  So on ``val_strain_only`` the strain-keyed
tables vanish and the model must lean on the chemical- and batch-keyed ones;
on ``val_chem_only`` the reverse.  That is the intended, and honest, behaviour.

Leakage control
---------------
Two separate mechanisms, both required:

1. **Nothing is fitted outside the fit mask.**  Protein statistics, every
   group-mean table and every categorical vocabulary come from ``fit_mask`` rows
   only.  Categories absent from the fit set map to code ``-1``
   (``__UNSEEN__``), never to a neighbouring level.

2. **Out-of-fold encoding for fit rows.**  A group mean used as a feature for a
   row that contributed to it is target leakage -- the row partly predicts
   itself, and the model then over-trusts the feature at evaluation time where
   no such self-contribution exists.  :class:`EncoderSet` therefore fits
   ``n_folds`` extra table sets, each excluding one fold, and serves fit rows
   from the set that excludes them.  Evaluation and test rows are served from
   the full-fit tables.  ``self_contribution_audit`` proves the difference is
   non-zero, i.e. that the OOF machinery is actually engaged.

The same class instantiated with a narrower ``fit_mask`` powers the inner
chemical-holdout used for hyper-parameter selection, so tuning never touches the
``val_*`` rows that are reported.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import CHEM_COL

# An all-NaN protein column (never detected in the fit rows) is a legitimate
# state: the statistic is genuinely undefined and NaN is the intended encoding.
for _msg in ("Mean of empty slice", "All-NaN slice encountered",
             "Degrees of freedom <= 0", "invalid value encountered"):
    warnings.filterwarnings("ignore", message=_msg)

__all__ = [
    "MIN_CELL_N",
    "N_FOLDS",
    "DELTA_TABLE_KEYS",
    "ABUND_TABLE_KEYS",
    "CAT_FEATURES",
    "NUM_FEATURES",
    "FEATURE_NAMES",
    "EncoderSet",
    "add_derived_columns",
]

#: Minimum finite fit observations in a ``(group, protein)`` cell before its
#: mean is emitted; thinner cells are NaN (never a one-observation "mean").
MIN_CELL_N = 3

#: Folds used for the out-of-fold encoding of fit rows.
N_FOLDS = 5

#: Group-mean ``Delta`` tables: ``name -> grouping keys``.
DELTA_TABLE_KEYS: dict[str, list[str]] = {
    "d_ctx":         ["Strains", "Medium", "Temperature", "pert_time"],
    "d_ctxb":        ["data_source", "Strains", "Medium", "Temperature", "pert_time"],
    "d_strain":      ["Strains"],
    "d_drug":        [CHEM_COL],
    "d_drug_time":   [CHEM_COL, "pert_time"],
    "d_drug_strain": [CHEM_COL, "Strains"],
    "d_batch":       ["data_source"],
    "d_batch_time":  ["data_source", "pert_time"],
    "d_plate":       ["Yeast_cell_plate"],
    "d_instr":       ["instrument"],
    "d_time":        ["pert_time"],
    "d_mtt":         ["Medium", "Temperature", "pert_time"],
}

#: Group-mean *absolute abundance* tables.  ``y_ctxb`` is the predictor that
#: scored 0.4454 in Step 2, supplied here as a single feature column.
ABUND_TABLE_KEYS: dict[str, list[str]] = {
    "y_ctxb": ["data_source", "Strains", "Medium", "Temperature", "pert_time"],
    "y_mtt":  ["Medium", "Temperature", "pert_time"],
}

#: Per-protein statistics, in emission order.
PROT_STATS = ["prot_mean_y", "prot_sd_y", "prot_detect",
              "prot_mean_d", "prot_sd_d", "prot_mean_absd"]

#: Categorical features (pandas ``category`` dtype; ``__UNSEEN__`` for levels
#: absent from the fit set).
CAT_FEATURES = ["data_source", "instrument", "Strains", "Medium",
                CHEM_COL, "Yeast_cell_plate", "pert_id"]

#: Numeric features, in emission order.
NUM_FEATURES = (
    ["c_ctrl", "c_ctrl_centered"]
    + list(DELTA_TABLE_KEYS)
    + list(ABUND_TABLE_KEYS)
    + PROT_STATS
    + ["pert_time_num", "temperature_num", "well_row", "well_col",
       "ctrl_median", "ctrl_detect", "has_drug_prior", "has_strain_prior"]
)

FEATURE_NAMES = NUM_FEATURES + CAT_FEATURES

# ---------------------------------------------------------------------------
# Availability regimes
# ---------------------------------------------------------------------------
# The OOD splits do not merely shift the feature *distribution*, they delete
# whole feature families: on a novel-strain split every strain-keyed table is
# NaN and ``Strains`` is out-of-vocabulary.  A model fitted on train rows -- where
# those features are present in ~98% of rows -- and then served such a row is
# pushed down a default branch almost no training data shaped.  Measured
# empirically in ``18_probe_feature_availability.py``, that mismatch cost 0.106
# of the S2 module score while the same model *beat* the benchmark on the splits
# where nothing is missing.
#
# The fix is to train one specialist per availability regime, masking exactly the
# features that regime cannot see.  Every specialist still trains on every train
# row, so no data is lost -- only the columns the regime would not have.
#
#: Features that require the *chemical* to have been seen in training.
CHEM_DEPENDENT = ["d_drug", "d_drug_time", "d_drug_strain", CHEM_COL]

#: Features that require the *strain* to have been seen in training.
STRAIN_DEPENDENT = ["d_ctx", "d_ctxb", "d_strain", "d_drug_strain", "y_ctxb", "Strains"]

#: regime name -> which dependency families are unavailable.  Routing at
#: prediction time is by entity novelty (is this strain / chemical in the fit
#: set?), not by split label, so the same routing works on the test set.
REGIMES: dict[str, tuple[str, ...]] = {
    "full": (),                            # both entities seen  -> val_time
    "chem_novel": ("chem",),               # S1: val_chem_only
    "strain_novel": ("strain",),           # S2: val_strain_only
    "both_novel": ("chem", "strain"),      # S3: val_both
}


def regime_for_rows(has_drug_prior: np.ndarray, has_strain_prior: np.ndarray) -> np.ndarray:
    """Map per-row entity-availability flags onto regime names."""
    d = np.asarray(has_drug_prior) > 0.5
    s = np.asarray(has_strain_prior) > 0.5
    out = np.empty(len(d), dtype=object)
    out[d & s] = "full"
    out[~d & s] = "chem_novel"
    out[d & ~s] = "strain_novel"
    out[~d & ~s] = "both_novel"
    return out


def apply_regime_mask(X: pd.DataFrame, regime: str, copy: bool = True) -> pd.DataFrame:
    """Blank the features a given availability regime cannot see.

    Masking a train row makes it indistinguishable from a genuine OOD row of that
    regime: the numeric group means become NaN, the entity categorical becomes
    ``__UNSEEN__``, and the corresponding prior-availability flag becomes 0 --
    exactly the state ``build_block`` produces for a real novel entity.

    Parameters
    ----------
    X : pandas.DataFrame
        Design matrix from :meth:`EncoderSet.build_block`.
    regime : str
        Key of :data:`REGIMES`.
    copy : bool
        Copy before masking.  ``False`` mutates in place to save memory when the
        caller owns ``X``.

    Returns
    -------
    pandas.DataFrame
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {list(REGIMES)}")
    out = X.copy() if copy else X
    families = REGIMES[regime]
    cols: list[str] = []
    if "chem" in families:
        cols += CHEM_DEPENDENT
        out["has_drug_prior"] = np.float32(0.0)
    if "strain" in families:
        cols += STRAIN_DEPENDENT
        out["has_strain_prior"] = np.float32(0.0)
    for c in dict.fromkeys(cols):
        if c in CAT_FEATURES:
            out[c] = pd.Categorical(np.repeat("__UNSEEN__", len(out)),
                                    categories=out[c].cat.categories)
        else:
            out[c] = np.float32(np.nan)
    return out


# ---------------------------------------------------------------------------
# Metadata preparation
# ---------------------------------------------------------------------------
def add_derived_columns(meta: pd.DataFrame) -> pd.DataFrame:
    """Add numeric / decomposed columns the feature builder needs.

    ``pert_time`` and ``Temperature`` become numeric (the plan requires
    ``pert_time`` be continuous so the tree can interpolate within the observed
    15-240 min grid), and ``protein_well`` is split into its row letter and
    column number so plate geometry is expressible as two ordered numbers
    rather than an 88-level categorical.

    Raises
    ------
    ValueError
        If any value fails to parse, rather than silently yielding NaN.
    """
    meta = meta.copy()
    for src, dst in [("pert_time", "pert_time_num"), ("Temperature", "temperature_num")]:
        v = pd.to_numeric(meta[src], errors="coerce")
        if v.isna().any():
            bad = sorted(set(meta.loc[v.isna(), src].astype(str)))[:5]
            raise ValueError(f"{src}: {int(v.isna().sum())} values are not numeric, e.g. {bad}")
        meta[dst] = v.astype("float32")

    w = meta["protein_well"].astype(str)
    row = w.str.extract(r"^([A-Ha-h])", expand=False)
    col = pd.to_numeric(w.str.extract(r"(\d+)$", expand=False), errors="coerce")
    if row.isna().any() or col.isna().any():
        bad = sorted(set(w[row.isna() | col.isna()]))[:5]
        raise ValueError(f"protein_well: unparsable wells, e.g. {bad}")
    meta["well_row"] = row.str.upper().map({c: i + 1 for i, c in enumerate("ABCDEFGH")}).astype("float32")
    meta["well_col"] = col.astype("float32")
    return meta


# ---------------------------------------------------------------------------
# Grouped NaN-skipping means
# ---------------------------------------------------------------------------
def _codes_for(meta: pd.DataFrame, keys: list[str],
               vocab: dict[tuple, int] | None = None,
               ) -> tuple[np.ndarray, dict[tuple, int]]:
    """Integer group codes for ``keys``; ``-1`` when the tuple is not in ``vocab``.

    A vocabulary built on the fit rows is reused for every other row set, so an
    unseen combination is explicitly out-of-vocabulary instead of colliding with
    a different group.
    """
    tup = list(map(tuple, meta[keys].astype(str).to_numpy())) if keys \
        else [("__global__",)] * len(meta)
    if vocab is None:
        vocab = {k: i for i, k in enumerate(dict.fromkeys(tup))}
    codes = np.fromiter((vocab.get(k, -1) for k in tup), dtype=np.int32, count=len(tup))
    return codes, vocab


def _grouped_nanmean(M: np.ndarray, codes: np.ndarray, n_groups: int,
                     min_cell_n: int = MIN_CELL_N) -> np.ndarray:
    """``(n_groups, n_proteins)`` NaN-skipping column means per group code.

    Implemented as two sparse-by-dense products (one for the sums, one for the
    finite counts), so the cost is two passes over ``M`` regardless of the
    number of groups -- a per-group boolean-index loop would re-read ``M`` once
    per group.  Accumulation is float64: summing ~5,000 float32 log2 values per
    cell loses several digits otherwise.

    Cells with fewer than ``min_cell_n`` finite observations are NaN.
    """
    n, p = M.shape
    fin = np.isfinite(M)
    keep = codes >= 0
    ind = sp.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float64),
         (codes[keep], np.flatnonzero(keep))),
        shape=(n_groups, n),
    )
    sums = ind @ np.where(fin, M, 0.0).astype(np.float64, copy=False)
    cnts = ind @ fin.astype(np.float64)
    out = np.full((n_groups, p), np.nan, dtype=np.float64)
    np.divide(sums, cnts, out=out, where=cnts > 0)
    out[cnts < min_cell_n] = np.nan
    return out.astype(np.float32)


@dataclass
class _Table:
    """One group-mean table plus the vocabulary that indexes it."""

    name: str
    keys: list[str]
    vocab: dict[tuple, int]
    means: np.ndarray                     # (n_groups, n_proteins) float32
    n_groups_total: int = 0
    n_cells_defined: float = 0.0

    def rows_for(self, codes: np.ndarray) -> np.ndarray:
        """``(len(codes), n_proteins)`` lookup; out-of-vocabulary rows are NaN."""
        out = np.full((len(codes), self.means.shape[1]), np.nan, dtype=np.float32)
        ok = codes >= 0
        if ok.any():
            out[ok] = self.means[codes[ok]]
        return out


@dataclass
class _TableSet:
    """A complete set of tables + protein statistics fitted on one row subset."""

    delta: dict[str, _Table] = field(default_factory=dict)
    abund: dict[str, _Table] = field(default_factory=dict)
    prot: dict[str, np.ndarray] = field(default_factory=dict)
    seen_chem: set = field(default_factory=set)
    seen_strain: set = field(default_factory=set)
    n_rows: int = 0


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class EncoderSet:
    """Train-fitted encoders for the long-format design matrix.

    Parameters
    ----------
    meta : pandas.DataFrame
        All treated rows, already passed through :func:`add_derived_columns`.
        Row order defines matrix row order.
    Y, D, C : numpy.ndarray
        ``(n, p)`` log2 abundance, true ``Delta`` and matched-control anchor.
    fit_mask : numpy.ndarray
        Boolean mask of the rows every statistic may be fitted on.
    n_folds : int
        Out-of-fold count for the fit rows' own encodings.
    seed : int
        Fold-assignment seed.
    verbose : bool
        Progress logging.

    Attributes
    ----------
    full : _TableSet
        Tables fitted on all ``fit_mask`` rows; used to serve non-fit rows.
    folds : list of _TableSet
        ``folds[k]`` excludes fold ``k``; used to serve fit rows in fold ``k``.
    fold_id : numpy.ndarray
        ``(n,)`` int32, fold index for fit rows and ``-1`` elsewhere.
    """

    def __init__(self, meta: pd.DataFrame, Y: np.ndarray, D: np.ndarray, C: np.ndarray,
                 fit_mask: np.ndarray, n_folds: int = N_FOLDS, seed: int = 42,
                 verbose: bool = True) -> None:
        self.meta = meta
        self.Y, self.D, self.C = Y, D, C
        self.fit_mask = np.asarray(fit_mask, dtype=bool)
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self.verbose = verbose
        self.n, self.p = D.shape

        # Row-level scalars that need no fitting.
        with np.errstate(all="ignore"):
            self.ctrl_median = np.nanmedian(C, axis=1).astype("float32")
        self.ctrl_detect = np.isfinite(C).mean(axis=1).astype("float32")

        # Deterministic fold assignment over fit rows only.
        rng = np.random.default_rng(self.seed)
        self.fold_id = np.full(self.n, -1, dtype=np.int32)
        fit_idx = np.flatnonzero(self.fit_mask)
        self.fold_id[fit_idx] = rng.permutation(len(fit_idx)) % self.n_folds

        self._log(f"fitting encoders on {len(fit_idx)} rows, {self.n_folds}-fold OOF")
        self.full = self._fit(self.fit_mask, "full")
        self.folds = [self._fit(self.fit_mask & (self.fold_id != k), f"oof{k}")
                      for k in range(self.n_folds)]

        # Categorical vocabularies come from the fit rows only.
        self.cat_levels: dict[str, list[str]] = {}
        for c in CAT_FEATURES:
            lv = sorted(set(meta.loc[self.fit_mask, c].astype(str)))
            self.cat_levels[c] = lv + ["__UNSEEN__"]
        self._log("categorical vocabularies: "
                  + ", ".join(f"{c}={len(v) - 1}(+UNSEEN)" for c, v in self.cat_levels.items()))

    # -- internals ---------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [enc] {msg}", flush=True)

    def _fit(self, mask: np.ndarray, label: str) -> _TableSet:
        """Fit every table and protein statistic on ``mask`` rows."""
        ts = _TableSet(n_rows=int(mask.sum()))
        sub = self.meta.loc[mask]
        Dm, Ym = self.D[mask], self.Y[mask]

        for name, keys in DELTA_TABLE_KEYS.items():
            codes, vocab = _codes_for(sub, keys)
            means = _grouped_nanmean(Dm, codes, len(vocab))
            ts.delta[name] = _Table(name, keys, vocab, means, len(vocab),
                                    float(np.isfinite(means).mean()))
        for name, keys in ABUND_TABLE_KEYS.items():
            codes, vocab = _codes_for(sub, keys)
            means = _grouped_nanmean(Ym, codes, len(vocab))
            ts.abund[name] = _Table(name, keys, vocab, means, len(vocab),
                                    float(np.isfinite(means).mean()))

        with np.errstate(all="ignore"):
            ts.prot = {
                "prot_mean_y": np.nanmean(Ym, axis=0).astype("float32"),
                "prot_sd_y": np.nanstd(Ym, axis=0).astype("float32"),
                "prot_detect": np.isfinite(Ym).mean(axis=0).astype("float32"),
                "prot_mean_d": np.nanmean(Dm, axis=0).astype("float32"),
                "prot_sd_d": np.nanstd(Dm, axis=0).astype("float32"),
                "prot_mean_absd": np.nanmean(np.abs(Dm), axis=0).astype("float32"),
            }
        ts.seen_chem = set(sub[CHEM_COL].astype(str))
        ts.seen_strain = set(sub["Strains"].astype(str))
        self._log(f"{label}: {ts.n_rows} rows | "
                  + " ".join(f"{k}={v.n_groups_total}g/{100 * v.n_cells_defined:.0f}%"
                             for k, v in ts.delta.items()))
        return ts

    def _tableset_for(self, sample_idx: np.ndarray) -> list[tuple[_TableSet, np.ndarray]]:
        """Partition ``sample_idx`` by which table set must serve it."""
        fid = self.fold_id[sample_idx]
        groups: list[tuple[_TableSet, np.ndarray]] = []
        non_fit = fid < 0
        if non_fit.any():
            groups.append((self.full, sample_idx[non_fit]))
        for k in range(self.n_folds):
            m = fid == k
            if m.any():
                groups.append((self.folds[k], sample_idx[m]))
        return groups

    # -- public ------------------------------------------------------------
    def build_block(self, sample_idx: np.ndarray, cell_mask: np.ndarray | None = None,
                    external: dict[str, object] | None = None,
                    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Long-format features for a block of samples.

        Parameters
        ----------
        sample_idx : numpy.ndarray
            Row indices into the source frame/matrices (``self`` or ``external``).
        cell_mask : numpy.ndarray, optional
            ``(len(sample_idx), n_proteins)`` boolean; only ``True`` cells are
            emitted.  ``None`` emits every cell (used at prediction time).
        external : dict, optional
            ``{"meta", "C", "Y", "D"}`` for a cohort the encoders were *not*
            fitted on -- the independent test set.  Every feature is then served
            from the full-fit tables (out-of-fold routing is meaningless for rows
            that never entered any fit), and the categorical vocabularies,
            protein statistics and group means all remain the train-fitted ones.
            ``Y`` and ``D`` are used only to return targets and may be ``None``.

        Returns
        -------
        X : pandas.DataFrame
            ``(n_rows, len(FEATURE_NAMES))`` design matrix, numeric columns
            float32 and categoricals ``category`` dtype.
        y_abs, y_delta : numpy.ndarray
            Targets for the two formulations (``Y`` and ``Delta``); NaN where
            unmeasured, all-NaN when ``external`` supplies no targets.
        row_sample : numpy.ndarray
            Index into ``sample_idx`` for every emitted row, so predictions can
            be scattered back into matrix form.
        """
        sample_idx = np.asarray(sample_idx)
        if external is None:
            meta_src, C_src = self.meta, self.C
            Y_src, D_src = self.Y, self.D
            cmed_src, cdet_src = self.ctrl_median, self.ctrl_detect
            servers = self._tableset_for(sample_idx)
        else:
            meta_src = external["meta"]
            C_src = external["C"]
            Y_src, D_src = external.get("Y"), external.get("D")
            with np.errstate(all="ignore"):
                cmed_src = np.nanmedian(C_src, axis=1).astype("float32")
            cdet_src = np.isfinite(C_src).mean(axis=1).astype("float32")
            servers = [(self.full, sample_idx)]

        nb, p = len(sample_idx), self.p
        keep = np.ones((nb, p), dtype=bool) if cell_mask is None else np.asarray(cell_mask, dtype=bool)
        if keep.shape != (nb, p):
            raise ValueError(f"cell_mask has shape {keep.shape}, expected {(nb, p)}")
        n_rows = int(keep.sum())
        pos = pd.Series(np.arange(nb), index=sample_idx)

        cols: dict[str, np.ndarray] = {}

        # --- cell-level: control anchor -----------------------------------
        Cb = C_src[sample_idx]
        cols["c_ctrl"] = Cb[keep]
        cols["c_ctrl_centered"] = (Cb - cmed_src[sample_idx][:, None])[keep]

        # --- cell-level: group-mean tables (per serving table set) --------
        for tname in list(DELTA_TABLE_KEYS) + list(ABUND_TABLE_KEYS):
            buf = np.empty((nb, p), dtype=np.float32)
            for ts, idx in servers:
                tab = ts.delta.get(tname) or ts.abund[tname]
                codes, _ = _codes_for(meta_src.iloc[idx], tab.keys, tab.vocab)
                buf[pos.loc[idx].to_numpy()] = tab.rows_for(codes)
            cols[tname] = buf[keep]
            del buf

        # --- protein-level -------------------------------------------------
        # Statistics are per-protein, so the serving table set matters here too.
        for sname in PROT_STATS:
            buf = np.empty((nb, p), dtype=np.float32)
            for ts, idx in servers:
                buf[pos.loc[idx].to_numpy()] = ts.prot[sname][None, :]
            cols[sname] = buf[keep]
            del buf

        # --- sample-level numerics (broadcast down the protein axis) ------
        sub = meta_src.iloc[sample_idx]
        rep = keep.sum(axis=1)
        for fname, vals in [
            ("pert_time_num", sub["pert_time_num"].to_numpy("float32")),
            ("temperature_num", sub["temperature_num"].to_numpy("float32")),
            ("well_row", sub["well_row"].to_numpy("float32")),
            ("well_col", sub["well_col"].to_numpy("float32")),
            ("ctrl_median", cmed_src[sample_idx]),
            ("ctrl_detect", cdet_src[sample_idx]),
        ]:
            cols[fname] = np.repeat(vals, rep)

        # Prior-availability flags, resolved against the *serving* fit set.
        has_drug = np.zeros(nb, dtype="float32")
        has_strain = np.zeros(nb, dtype="float32")
        for ts, idx in servers:
            loc = pos.loc[idx].to_numpy()
            m = meta_src.iloc[idx]
            has_drug[loc] = m[CHEM_COL].astype(str).isin(ts.seen_chem).to_numpy("float32")
            has_strain[loc] = m["Strains"].astype(str).isin(ts.seen_strain).to_numpy("float32")
        cols["has_drug_prior"] = np.repeat(has_drug, rep)
        cols["has_strain_prior"] = np.repeat(has_strain, rep)

        X = pd.DataFrame({k: np.asarray(cols[k], dtype="float32") for k in NUM_FEATURES},
                         copy=False)

        # --- sample-level categoricals ------------------------------------
        for c in CAT_FEATURES:
            lv = self.cat_levels[c]
            v = sub[c].astype(str).where(sub[c].astype(str).isin(lv[:-1]), "__UNSEEN__")
            X[c] = pd.Categorical(np.repeat(v.to_numpy(), rep), categories=lv)

        y_abs = Y_src[sample_idx][keep] if Y_src is not None \
            else np.full(n_rows, np.nan, dtype="float32")
        y_delta = D_src[sample_idx][keep] if D_src is not None \
            else np.full(n_rows, np.nan, dtype="float32")
        row_sample = np.repeat(np.arange(nb), rep)
        if len(X) != n_rows:
            raise ValueError(f"built {len(X)} rows, expected {n_rows}")
        return X[FEATURE_NAMES], y_abs, y_delta, row_sample

    # -- diagnostics -------------------------------------------------------
    def self_contribution_audit(self, n_samples: int = 40) -> dict[str, object]:
        """Quantify how much a fit row's own ``Delta`` moves its own encoding.

        Compares the OOF value actually served to a fit row against the
        full-fit value.  A non-zero difference proves the OOF path is engaged;
        the magnitude is the target leakage that would otherwise be baked into
        training.  Reported per table, in ``Delta`` (log2) units.
        """
        rng = np.random.default_rng(self.seed)
        idx = np.flatnonzero(self.fit_mask)
        idx = rng.choice(idx, size=min(n_samples, len(idx)), replace=False)
        out: dict[str, object] = {"n_samples_audited": int(len(idx))}
        for tname, keys in DELTA_TABLE_KEYS.items():
            diffs = []
            for i in idx:
                k = int(self.fold_id[i])
                sub_i = self.meta.iloc[[i]]
                c_full, _ = _codes_for(sub_i, keys, self.full.delta[tname].vocab)
                c_oof, _ = _codes_for(sub_i, keys, self.folds[k].delta[tname].vocab)
                a = self.full.delta[tname].rows_for(c_full)[0]
                b = self.folds[k].delta[tname].rows_for(c_oof)[0]
                m = np.isfinite(a) & np.isfinite(b)
                if m.any():
                    diffs.append(float(np.mean(np.abs(a[m] - b[m]))))
            out[tname] = {
                "mean_abs_oof_minus_full": round(float(np.mean(diffs)), 6) if diffs else None,
                "max_abs_oof_minus_full": round(float(np.max(diffs)), 6) if diffs else None,
                "n_groups_full": self.full.delta[tname].n_groups_total,
            }
        return out

    def coverage_report(self, masks: dict[str, np.ndarray]) -> pd.DataFrame:
        """Per-split fraction of *defined* cells for every group-mean table.

        This is the table that explains the split-specific score profile: a
        strain-keyed feature is 0% covered on a novel-strain split, so whatever
        the model achieves there comes from the remaining feature families.
        """
        rows = []
        for split, m in masks.items():
            idx = np.flatnonzero(m)
            if not len(idx):
                continue
            take = idx[:: max(1, len(idx) // 200)]          # bounded probe
            for tname in list(DELTA_TABLE_KEYS) + list(ABUND_TABLE_KEYS):
                frac, wt = [], []
                for ts, sidx in self._tableset_for(take):
                    tab = ts.delta.get(tname) or ts.abund[tname]
                    codes, _ = _codes_for(self.meta.iloc[sidx], tab.keys, tab.vocab)
                    frac.append(np.isfinite(tab.rows_for(codes)).mean())
                    wt.append(len(sidx))
                rows.append({"split": split, "feature": tname,
                             "frac_cells_defined": round(float(np.average(frac, weights=wt)), 6),
                             "n_samples_probed": int(len(take))})
        return pd.DataFrame(rows)


def assemble_training_matrix(enc: EncoderSet, sample_idx: np.ndarray,
                             cell_mask_fn, chunk: int = 300, label: str = "train",
                             ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build a long design matrix over ``sample_idx`` in memory-bounded chunks.

    Parameters
    ----------
    enc : EncoderSet
        Fitted encoders.
    sample_idx : numpy.ndarray
        Rows to emit.
    cell_mask_fn : callable
        ``(sample_idx_block) -> (len(block), n_proteins)`` boolean mask, used to
        keep only finite-target and subsampled cells.
    chunk : int
        Samples per block.  300 x 5,243 x ~36 float32 is ~0.8 GB peak.
    label : str
        Progress-log label.

    Returns
    -------
    X, y_abs, y_delta
    """
    Xs, ya, yd = [], [], []
    t0 = time.time()
    for s in range(0, len(sample_idx), chunk):
        blk = sample_idx[s:s + chunk]
        X, a, d, _ = enc.build_block(blk, cell_mask_fn(blk))
        Xs.append(X)
        ya.append(a)
        yd.append(d)
        done = min(s + chunk, len(sample_idx))
        print(f"  [{label}] {done}/{len(sample_idx)} samples | "
              f"{sum(len(x) for x in Xs):,} rows | {time.time() - t0:.0f}s", flush=True)
    X = pd.concat(Xs, ignore_index=True, copy=False)
    return X, np.concatenate(ya), np.concatenate(yd)
