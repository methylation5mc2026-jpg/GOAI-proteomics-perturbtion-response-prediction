"""
Shared paths, constants and loaders for the WAYB/WAYC proteomics QC pipeline.

Design decisions encoded here (see README.md for rationale):

* Missingness in the released matrices is NaN-only (no exact zeros were observed),
  so NaN is treated as "not detected / below detection" and is never silently
  conflated with a true zero.
* Values are linear MS intensities and are log2-transformed as log2(x) with NaN
  preserved. No imputation is applied to the exported log2 matrices.
* 'perturbation_no_concentration' is the chemical entity key. 'pert_id' is a
  batch-local plate/dose slot code and is NOT globally unique per chemical
  (verified in 03_refine_matching.py), so it is never used as an entity key.
* Any statistic that must be "frozen" per the competition rules (normalisation
  offsets, imputation means, PCA loadings) is fitted on the 'train' split of
  train_val ONLY and then applied unchanged to every other split and to test.
"""

from __future__ import annotations

from repo_paths import (
    DATA_DIR,
    FIGURES_DIR,
    INPUT_DIR,
    LOGS_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    WORKFLOW_DIR,
)

from pathlib import Path

import numpy as np
import pandas as pd

SESSION = REPO_ROOT
INPUT = INPUT_DIR
DATA = DATA_DIR
RESULTS = RESULTS_DIR
FIGURES = FIGURES_DIR
WORKFLOW = WORKFLOW_DIR
LOGS = LOGS_DIR

_INPUT_CANDIDATES = {
    "metadata_train_val": (
        "WAYB_WAYC_metadata_train_val.csv",
        "WAYB_WAYC_metadata_train_val(1).csv",
    ),
    "metadata_test": (
        "WAYB_WAYC_metadata_test.csv",
        "WAYB_WAYC_metadata_test(1).csv",
    ),
    "proteome_train_val": ("WAYB_WAYC_proteome_raw_train_val.csv",),
    "proteome_test": ("WAYB_WAYC_proteome_raw_test.csv",),
}


def _selected_input(logical_name: str) -> Path:
    """Return the first available filename for one logical competition input."""
    names = _INPUT_CANDIDATES[logical_name]
    return next((INPUT / name for name in names if (INPUT / name).is_file()),
                INPUT / names[0])


def require_input_files(*logical_names: str) -> None:
    """Fail with all accepted filenames when required competition data are absent."""
    missing = []
    for logical_name in logical_names:
        names = _INPUT_CANDIDATES[logical_name]
        if not any((INPUT / name).is_file() for name in names):
            missing.append(f"{logical_name}: " + " or ".join(names))
    if missing:
        raise FileNotFoundError(
            f"Missing competition input files under {INPUT}. Accepted names:\n  "
            + "\n  ".join(missing)
        )


META_TV = _selected_input("metadata_train_val")
META_TE = _selected_input("metadata_test")
PROT_TV = _selected_input("proteome_train_val")
PROT_TE = _selected_input("proteome_test")

ID_COL = "sample_ID"
CHEM_COL = "perturbation_no_concentration"

CONTROL_CHEMS = ("Water", "DMSO")
QC_CHEMS = ("Quality Control",)

#: Metadata columns describing biological condition vs measurement context.
BIO_COLS = ["Strains", "Medium", "Temperature", "pert_time", CHEM_COL]
CTX_COLS = ["data_source", "instrument", "Yeast_cell_plate", "protein_well"]

#: Control-matching fallback hierarchy, finest -> coarsest. Frozen rule.
CONTEXT_LEVELS: list[tuple[str, list[str]]] = [
    ("L1_full_ctx", ["data_source", "Strains", "Medium", "Temperature", "pert_time"]),
    ("L2_drop_time", ["data_source", "Strains", "Medium", "Temperature"]),
    ("L3_drop_batch", ["Strains", "Medium", "Temperature", "pert_time"]),
    ("L4_strain_media", ["Strains", "Medium", "Temperature"]),
    ("L5_strain", ["Strains"]),
    ("L6_global", []),
]

SEED = 42


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train_val and test metadata as strings (no dtype coercion)."""
    require_input_files("metadata_train_val", "metadata_test")
    tv = pd.read_csv(META_TV, dtype=str)
    te = pd.read_csv(META_TE, dtype=str)
    return tv, te


def load_proteome(path: Path, label: str) -> tuple[pd.Series, np.ndarray, list[str]]:
    """Load a proteome CSV into a float32 matrix.

    Parameters
    ----------
    path : pathlib.Path
        Path to the proteome CSV (first column is ``sample_ID``).
    label : str
        Label used for progress logging.

    Returns
    -------
    ids : pandas.Series
        Sample identifiers, in file order.
    mat : numpy.ndarray
        ``(n_samples, n_proteins)`` float32 matrix of linear intensities with
        NaN for non-detected values.
    proteins : list of str
        Protein column names, in file order.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing proteome input for {label}: {path}")
    print(f"  [load] reading {label} from {path.name} ...", flush=True)
    df = pd.read_csv(path)
    ids = df[ID_COL].astype(str)
    proteins = [c for c in df.columns if c != ID_COL]
    mat = df[proteins].to_numpy(dtype="float32", na_value=np.nan)
    print(f"  [load] {label}: {mat.shape[0]} samples x {mat.shape[1]} proteins "
          f"({mat.nbytes / 1e6:.0f} MB float32)", flush=True)
    return ids, mat, proteins


def align_to_metadata(ids: pd.Series, mat: np.ndarray,
                      meta: pd.DataFrame, label: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Reorder a proteome matrix to match metadata row order, verifying identity.

    Raises
    ------
    ValueError
        If the sample_ID sets differ between metadata and proteome matrix.
    """
    m_ids = meta[ID_COL].astype(str)
    s_meta, s_prot = set(m_ids), set(ids)
    if s_meta != s_prot:
        raise ValueError(
            f"{label}: sample_ID mismatch - meta_only={len(s_meta - s_prot)}, "
            f"prot_only={len(s_prot - s_meta)}"
        )
    pos = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
    order = pos.reindex(m_ids.to_numpy()).to_numpy()
    print(f"  [align] {label}: sample_ID sets identical (n={len(m_ids)}); "
          f"already in same order = {bool((order == np.arange(len(order))).all())}",
          flush=True)
    return meta.reset_index(drop=True), mat[order]


def log2_transform(mat: np.ndarray) -> np.ndarray:
    """log2-transform linear intensities, preserving NaN and mapping <=0 to NaN."""
    out = mat.astype("float32", copy=True)
    nonpos = np.isfinite(out) & (out <= 0)
    n_nonpos = int(nonpos.sum())
    if n_nonpos:
        print(f"  [log2] {n_nonpos} non-positive values set to NaN before log2",
              flush=True)
        out[nonpos] = np.nan
    return np.log2(out, where=np.isfinite(out), out=np.full_like(out, np.nan))


def sample_role(chem: pd.Series) -> pd.Series:
    """Classify each sample as 'control', 'qc' or 'treatment' by chemical name."""
    role = pd.Series("treatment", index=chem.index, dtype=object)
    role[chem.isin(CONTROL_CHEMS)] = "control"
    role[chem.isin(QC_CHEMS)] = "qc"
    return role


def save_matrix_parquet(ids: pd.Series, mat: np.ndarray, proteins: list[str],
                        dest: Path, extra: pd.DataFrame | None = None) -> None:
    """Write a sample x protein matrix to parquet with sample_ID as first column."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(mat, columns=proteins)
    df.insert(0, ID_COL, np.asarray(ids, dtype=object))
    if extra is not None:
        for j, col in enumerate(extra.columns):
            df.insert(1 + j, col, extra[col].to_numpy())
    df.to_parquet(dest, index=False, compression="snappy")
    print(f"  [save] {dest.name}: {df.shape[0]} x {df.shape[1]} "
          f"-> {dest.stat().st_size / 1e6:.1f} MB", flush=True)
