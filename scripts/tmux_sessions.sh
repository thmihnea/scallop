#!/usr/bin/env bash
# Start one detached tmux session per run, in the repo root, with the flowmaps conda env active.
#   ./scripts/tmux_sessions.sh [session ...]   (default: the 8 ace_a_nme run sessions)
# Run ./setup.sh first: flowmaps is activated from the conda it used (saved in .conda_base).
# CONDA_BASE=<dir> overrides; without either, the conda on PATH is used.
# Existing sessions are left untouched. Attach with: tmux attach -t <session>
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
conda_base=${CONDA_BASE:-$(cat "$repo/.conda_base" 2>/dev/null || conda info --base 2>/dev/null || true)}
[[ -f $conda_base/etc/profile.d/conda.sh ]] || { echo "conda not found: run ./setup.sh first (or set CONDA_BASE)" >&2; exit 1; }
tmux=$(command -v tmux || echo "$conda_base/bin/tmux")
sessions=("$@"); (( $# )) || sessions=({square,triangle}{,-weight}-{cycle,no-cycle})
for s in "${sessions[@]}"; do
    if "$tmux" has-session -t "=$s" 2>/dev/null; then echo "$s: exists, skipped"; continue; fi
    "$tmux" new-session -d -s "$s" -c "$repo"
    "$tmux" send-keys -t "$s" "source $conda_base/etc/profile.d/conda.sh && conda activate flowmaps" Enter
    echo "$s: created"
done
