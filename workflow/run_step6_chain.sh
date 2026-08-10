#!/usr/bin/env bash
# Step 6 chain: cross-attention member -> hierarchical stacking -> evaluation/export.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1
PY=.venv/bin/python
LOG=logs/step6_chain.log
: > "$LOG"
echo "=== [1/3] 36_dynamic_cross_attention_gnn.py (epochs=300, checkpoints=150) ===" | tee -a "$LOG"
$PY workflow/36_dynamic_cross_attention_gnn.py --epochs 300 --checkpoints 150 \
    > logs/step6_36_xattn.log 2>&1
echo "  exit=$? at $(date -u +%H:%M:%S)" | tee -a "$LOG"
echo "=== [2/3] 37_hierarchical_cluster_stacking.py ===" | tee -a "$LOG"
$PY workflow/37_hierarchical_cluster_stacking.py --budget 420 --n-boot 1000 \
    > logs/step6_37_stack.log 2>&1
echo "  exit=$? at $(date -u +%H:%M:%S)" | tee -a "$LOG"
echo "=== [3/3] 38_eval_step6.py ===" | tee -a "$LOG"
$PY workflow/38_eval_step6.py > logs/step6_38_eval.log 2>&1
echo "  exit=$? at $(date -u +%H:%M:%S)" | tee -a "$LOG"
echo "=== chain complete ===" | tee -a "$LOG"
