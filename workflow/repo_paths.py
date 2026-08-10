"""Repository-relative paths shared by every workflow stage."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / "workflow"
INPUT_DIR = REPO_ROOT / "input"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
LOGS_DIR = REPO_ROOT / "logs"
