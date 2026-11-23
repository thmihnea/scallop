#!/usr/bin/env bash
# New-machine setup, following the README: installs Miniforge if conda is missing, creates the
# flowmaps env from environment.yml (updates it if it exists), and installs tmux if it is missing.
#   ./setup.sh                      then: ./scripts/tmux_sessions.sh
# CONDA_BASE=<dir> picks the conda install; by default the one on PATH, else a new ~/miniforge3.
# The conda used is saved in .conda_base, and tmux_sessions.sh activates flowmaps from it.
set -eo pipefail
cd "$(dirname "$0")"
conda_base=${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")}
if [[ ! -f $conda_base/etc/profile.d/conda.sh ]]; then
    echo "conda not found: installing Miniforge into $conda_base (set CONDA_BASE to use another)"
    installer=$(mktemp -d)/miniforge.sh
    url=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
    curl -fsSL -o "$installer" "$url" || wget -qO "$installer" "$url"
    bash "$installer" -b -p "$conda_base"
fi
source "$conda_base/etc/profile.d/conda.sh"
if conda env list | awk '$1 == "flowmaps" {found = 1} END {exit !found}'; then
    conda env update -n flowmaps -f environment.yml
else
    conda env create -f environment.yml
fi
command -v tmux >/dev/null || [[ -x $conda_base/bin/tmux ]] || conda install -y -n base -c conda-forge tmux
echo "$conda_base" > .conda_base
conda activate flowmaps
python -c "import torch; assert hasattr(torch.optim, 'Muon'), 'torch has no optim.Muon'; \
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "flowmaps is ready (conda: $conda_base). Next: ./scripts/tmux_sessions.sh"
