#!/usr/bin/env bash
# Start one detached tmux session per run, in the repo root, with the flowmaps conda env active.
#   bash scripts/tmux_sessions.sh [session ...]   (default: the 8 ace_a_nme run sessions)
# Existing sessions are left untouched. Attach with: tmux attach -t <session>
# Set CONDA_BASE if conda is not on PATH and not in /scratch2/mtm67/miniforge3.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
conda_base=${CONDA_BASE:-$(conda info --base 2>/dev/null || echo /scratch2/mtm67/miniforge3)}
[[ -f $conda_base/etc/profile.d/conda.sh ]] || { echo "no conda at $conda_base; set CONDA_BASE" >&2; exit 1; }
sessions=("$@"); (( $# )) || sessions=({square,triangle}{,-weight}-{cycle,no-cycle})
for s in "${sessions[@]}"; do
    if tmux has-session -t "=$s" 2>/dev/null; then echo "$s: exists, skipped"; continue; fi
    tmux new-session -d -s "$s" -c "$repo"
    tmux send-keys -t "$s" "source $conda_base/etc/profile.d/conda.sh && conda activate flowmaps" Enter
    echo "$s: created"
done
