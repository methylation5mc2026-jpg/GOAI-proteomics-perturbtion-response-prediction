#!/bin/bash
# Follow-on driver: waits for run_step5_chain.sh to finish, then runs the
# protein-cluster stacking ladder and the agentic self-evolution loop / test export.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

chain1_alive() {
    local d c
    for d in /proc/[0-9]*; do
        c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
        case "$c" in
            *run_step5_chain.sh*) return 0 ;;
            */python3*29_lcgo_oof_matrix.py*) return 0 ;;
            */python3*30_gnn_and_cluster_stacking.py*) return 0 ;;
        esac
    done
    return 1
}

echo "[chain2] waiting for chain 1 (valtest -> gnn -> dl cross-fit) ..."
while chain1_alive; do sleep 30; done
echo "[chain2] chain 1 is no longer running; last lines:"
tail -6 logs/step5_chain.log

# The role pool is passed in FULL rather than probed from disk. load_oof_cohort
# and load_val_cohort in script 30 already drop a role whose cache is absent and
# log the drop, so a member that failed to cross-fit is reported rather than
# silently becoming a zero column. An earlier version of this script probed for
# data/step5_cache/gnn_delta_oof.npy -- a file that never exists, because the real
# cache names are oof_gnn.npy / oof_dl.npy (see ROLES5 in script 30). That probe
# would have dropped BOTH neural members from every stacking run while printing a
# role pool that looked deliberate.
ROLES="gbdt_tab,gbdt_mol,gbdt_mol3d,bench,gnn,dl"
echo "[chain2] requesting role pool: $ROLES (absent caches are dropped with a log line)"
echo "[chain2] caches present in data/step5_cache:"
find data/step5_cache -mindepth 1 -maxdepth 1 -printf '    %f\n' 2>/dev/null | sort

echo "[chain2] === stage stack: protein-cluster non-negative stacking ladder ==="
uv run python workflow/30_gnn_and_cluster_stacking.py --stage stack \
    --roles "$ROLES" --k-ladder k4,k8,k12,k16 --alphas 0.0,0.02 \
    --budget 450 --n-boot 1000 > logs/step5_stack.log 2>&1
rc=$?
echo "[chain2] stack exit=$rc"
grep -E "dropped|HEADLINE|headline|rung|TOTAL|beats" logs/step5_stack.log | tail -30
if [ $rc -ne 0 ]; then tail -30 logs/step5_stack.log; exit $rc; fi

echo "[chain2] === stage loop: agentic self-evolution, test export, verification ==="
uv run python workflow/31_agentic_loop_runner.py \
    --iterations 12 --iter-budget 90 --start-cluster-col k8 > logs/step5_loop.log 2>&1
rc=$?
echo "[chain2] loop exit=$rc"
tail -30 logs/step5_loop.log
exit $rc
