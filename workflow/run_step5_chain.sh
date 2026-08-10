#!/bin/bash
# Step 5 driver: waits for the LCGO out-of-fold stage to finish, then runs the
# remaining long stages back to back so the machine is never idle.
#
# Stages are chained rather than parallelised on purpose: the LightGBM fits and
# the deep-learning featuriser both want the CPU, and running them together is
# what made a Step-4 fit take 1527 s instead of 37 s.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1
LOG=logs/step5_lcgo_oof.log

# This image ships no pgrep and no ps, so liveness is checked by scanning /proc
# cmdlines. The first version of this script used pgrep, silently found nothing,
# and declared a healthy job dead on the first tick -- hence the explicit scan.
oof_alive() {
    local d c
    for d in /proc/[0-9]*; do
        c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
        case "$c" in
            */python3*29_lcgo_oof_matrix.py*--stage*oof*) return 0 ;;
        esac
    done
    return 1
}

echo "[chain] waiting for the LCGO oof stage to complete ..."
while true; do
    if grep -q "=== stage oof complete ===" "$LOG" 2>/dev/null; then
        echo "[chain] oof stage complete"
        break
    fi
    if ! oof_alive; then
        echo "[chain] !! the oof process is gone and the completion marker is absent"
        tail -20 "$LOG"
        exit 1
    fi
    sleep 30
done

echo "[chain] === stage valtest: full-train counterparts for val and test ==="
uv run python workflow/29_lcgo_oof_matrix.py --stage valtest \
    --families lgb_tab,lgb_mol,lgb_mol3d > logs/step5_valtest.log 2>&1
rc=$?
echo "[chain] valtest exit=$rc"
tail -5 logs/step5_valtest.log
if [ $rc -ne 0 ]; then exit $rc; fi

echo "[chain] === stage gnn: graph-attention ResNet, cross-fitted ==="
uv run python workflow/30_gnn_and_cluster_stacking.py --stage gnn --arch gnn \
    --epochs 150 --lambda-graph 0.05 > logs/step5_gnn.log 2>&1
rc=$?
echo "[chain] gnn exit=$rc"
tail -8 logs/step5_gnn.log
if [ $rc -ne 0 ]; then exit $rc; fi

# Also cross-fit the Step-4 dense-head architecture. Step 4's stacking gave its
# deep member large weight, so having BOTH deep members available keeps the
# Step-4 -> Step-5 comparison about the weight space rather than about which
# deep architecture happened to be in the pool. The meta-learner is non-negative,
# so a weaker member simply receives ~0 weight.
echo "[chain] === stage dl: Step-4 dense-head architecture, cross-fitted ==="
uv run python workflow/30_gnn_and_cluster_stacking.py --stage gnn --arch mlp \
    --epochs 150 --skip-control > logs/step5_dl_crossfit.log 2>&1
rc=$?
echo "[chain] dl cross-fit exit=$rc"
tail -8 logs/step5_dl_crossfit.log
exit $rc
