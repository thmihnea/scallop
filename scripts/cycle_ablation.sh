#!/usr/bin/env bash
# Cycle-loss ablation for SCALLOP on ace_a_nme (ALA-2), from a bare GPU machine to a report.
#
#   bash cycle_ablation.sh            set up, train and evaluate in the background, then follow the log
#   bash cycle_ablation.sh status     latest progress table (also reports whether the job is alive)
#   bash cycle_ablation.sh logs       follow the main log again
#   bash cycle_ablation.sh report     print the comparison report
#   bash cycle_ablation.sh stop       stop gracefully; training saves checkpoints. Rerun to resume.
#
# Steps: clone the invertibility branch, build a Python 3.11 env with uv (no root needed),
# download the ace_a_nme data, check the CUDA-graph path on GPU 0, then
# py/launchers/cycle_ablation.py trains one velocity teacher and four SCALLOP runs from it.
# RECIPE=paper (default) uses the paper's stated setup for the teacher and every run: AdamW
# at 1e-4, 2.5% warm-up then sqrt decay, batch 1000, EMA 0.999, learned e^{-w} loss weights,
# times in [0, 1 - 1e-5], 6 shared + 3 density blocks, 4e6 updates per stage; the cycle runs
# stop the forward-in-time jump's gradient and weight the cycle term with its own e^{-w_c}.
# RECIPE=repo reproduces the first ablation (README recipe, 400k + 100k steps).
#
#   run1_square         (s, t) uniform on [0, 1]^2
#   run2_forward        (s, t) uniform on the triangle (notes.tex: t < s), the recipe default
#   run3_square_cycle   run1 + --cycle_loss
#   run4_forward_cycle  run2 + --cycle_loss
#
# then evaluates each with recipe v1 plus a round-trip check and writes
# $WORKDIR/ablation/report.md. Rerunning resumes every stage from its latest checkpoint.
# Until training has started, a rerun also updates the code to the branch head.
#
# Needs: Linux, an NVIDIA driver (nvidia-smi), git, curl, about 30 GB of disk.
# Settings (environment variables):
#   RECIPE        paper or repo                         (default: paper)
#   WORKDIR       where everything lives                (default: $HOME/scallop-paper-ablation;
#                                                        RECIPE=repo: $HOME/scallop-cycle-ablation)
#   TEACHER_STEPS / DISTILL_STEPS  updates per stage    (default: the recipe's; paper 4000000 each)
#   REPO_URL      repository to clone                   (default: https://github.com/thmihnea/scallop.git)
#   BRANCH        branch to clone                       (default: feat/invertibility_losses)
#   COMMIT        pin a commit                          (default: branch head at first launch)
#   TORCH_CUDA    torch wheel flavour, cu126 or cu128   (default: cu128 for compute capability >= 10, else cu126)
#   GPUS          number of GPUs to use, 0..GPUS-1      (default: all)
#   RUNS_PER_GPU  concurrent runs per GPU               (default: all four at once, lowered automatically on OOM)
#   RUN_TESTS     1 to run the CUDA-graph tests (about a minute) first (default: 1)
#   FOREGROUND    1 to run attached instead of in the background (default: 0)
set -Eeuo pipefail

RECIPE=${RECIPE:-paper}
[[ $RECIPE == paper || $RECIPE == repo ]] || { echo "RECIPE must be paper or repo"; exit 1; }
WORKDIR=${WORKDIR:-$HOME/$([[ $RECIPE == repo ]] && echo scallop-cycle-ablation || echo scallop-paper-ablation)}
TEACHER_STEPS=${TEACHER_STEPS:-0}
DISTILL_STEPS=${DISTILL_STEPS:-0}
REPO_URL=${REPO_URL:-https://github.com/thmihnea/scallop.git}
BRANCH=${BRANCH:-feat/invertibility_losses}
COMMIT=${COMMIT:-}
TORCH_CUDA=${TORCH_CUDA:-auto}
GPUS=${GPUS:-0}
RUNS_PER_GPU=${RUNS_PER_GPU:-0}
RUN_TESTS=${RUN_TESTS:-1}
FOREGROUND=${FOREGROUND:-0}

REPO=$WORKDIR/scallop
VENV=$WORKDIR/venv
PY=$VENV/bin/python
ROOT=$WORKDIR/ablation
LOGS=$WORKDIR/logs
LOG=$LOGS/ablation.log
LOCK=$WORKDIR/.lock
ORCHESTRATOR=(py/launchers/cycle_ablation.py)

say() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }
die() { say "ERROR: $*"; exit 1; }
trap 'say "ERROR: line $LINENO: $BASH_COMMAND (exit $?)"' ERR

mkdir -p "$WORKDIR" "$LOGS"
command=${1:-start}

case $command in
  status)
    [[ -x $PY ]] || die "nothing set up yet in $WORKDIR"
    cd "$REPO" && "$PY" "${ORCHESTRATOR[@]}" status --root "$ROOT"
    echo "--- last lines of $LOG"; tail -n 5 "$LOG" 2>/dev/null || true
    exit 0 ;;
  logs) exec tail -n 100 -F "$LOG" ;;
  report)
    cd "$REPO" && exec "$PY" "${ORCHESTRATOR[@]}" report --root "$ROOT" ;;
  stop)
    # pid files, not process names: the script may be saved under any name.
    # alive PIDFILE PATTERN: the pid in PIDFILE if that process is alive and its command matches.
    alive() {
      local pid
      pid=$(cat "$1" 2> /dev/null) || return 1
      [[ -n $pid ]] && tr '\0' ' ' 2> /dev/null < "/proc/$pid/cmdline" | grep -q -- "$2" && echo "$pid"
    }
    if leader=$(alive "$WORKDIR/job.pid" " start"); then
      # The detached job leads its own session: signal its process group. The orchestrator
      # then stops its training processes, which save a checkpoint first.
      kill -TERM -- "-$leader" 2> /dev/null || kill -TERM "$leader"
    elif orchestrator=$(alive "$ROOT/orchestrator.pid" "cycle_ablation.py"); then
      kill -TERM "$orchestrator"
    else
      say "not running"
      exit 0
    fi
    say "stopping; running stages save a checkpoint first (this can take a few minutes). Rerun to resume."
    exit 0 ;;
  start) ;;
  *) die "unknown command '$command' (start | status | logs | report | stop)" ;;
esac

# Detach so the run survives a closed SSH session; Ctrl-C only stops following the log.
if [[ -z ${ABLATION_DETACHED:-} && $FOREGROUND != 1 ]]; then
  if ! flock -n "$LOCK" true; then
    say "an ablation is already running in $WORKDIR; following its log (Ctrl-C to stop following)"
    exec tail -n 50 -F "$LOG"
  fi
  ABLATION_DETACHED=1 nohup setsid bash "$(readlink -f "${BASH_SOURCE[0]}")" start >> "$LOG" 2>&1 < /dev/null &
  say "started in the background (pid $!). Log: $LOG"
  say "Ctrl-C stops following the log, not the job. Later: bash $0 status | logs | report | stop"
  sleep 2
  exec tail -n 20 -F "$LOG"
fi
exec 9> "$LOCK"
flock -n 9 || die "another ablation holds $LOCK"
echo $$ > "$WORKDIR/job.pid"

say "===== cycle-loss ablation: setup on $(hostname) ====="

# ---------------------------------------------------------------- preflight
for tool in git curl nvidia-smi flock; do
  command -v "$tool" > /dev/null || die "'$tool' is required"
done
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader \
  | while read -r line; do say "GPU: $line"; done
n_gpus=$(nvidia-smi -L | grep -c '^GPU ')
(( n_gpus > 0 )) || die "no GPU visible to nvidia-smi"
say "CPUs: $(nproc) | RAM: $(free -g | awk '/^Mem:/ {print $2}') GB"
free_gb=$(df -P -BG "$WORKDIR" | awk 'NR == 2 {gsub("G", "", $4); print $4}')
say "free disk in $WORKDIR: ${free_gb} GB"
(( free_gb >= 30 )) || [[ -d $ROOT ]] || die "need about 30 GB free in $WORKDIR (have ${free_gb} GB)"

# ---------------------------------------------------------------- code
if [[ ! -d $REPO/.git ]]; then
  say "cloning $REPO_URL ($BRANCH)"
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO"
  [[ -z $COMMIT ]] || git -C "$REPO" checkout --detach "$COMMIT"
  git -C "$REPO" rev-parse HEAD > "$WORKDIR/commit.txt"
elif [[ ! -d $ROOT/teacher && -z $COMMIT ]]; then
  # Nothing trained yet, so take the latest code; afterwards the commit stays pinned.
  say "no training yet: updating the code to the head of $BRANCH"
  git -C "$REPO" fetch --quiet origin "$BRANCH"
  git -C "$REPO" checkout --quiet --detach FETCH_HEAD
  git -C "$REPO" rev-parse HEAD > "$WORKDIR/commit.txt"
fi
[[ -f $WORKDIR/commit.txt ]] || git -C "$REPO" rev-parse HEAD > "$WORKDIR/commit.txt"
commit=$(git -C "$REPO" rev-parse HEAD)
[[ $commit == "$(cat "$WORKDIR/commit.txt")" ]] \
  || die "repo HEAD $commit differs from the commit this ablation started with ($(cat "$WORKDIR/commit.txt")); use a new WORKDIR for new code"
say "code: $REPO_URL @ $commit"
cd "$REPO"

# ---------------------------------------------------------------- environment
if [[ $TORCH_CUDA == auto ]]; then
  major=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | cut -d. -f1 | sort -n | tail -1)
  TORCH_CUDA=$([[ ${major:-0} -ge 10 ]] && echo cu128 || echo cu126)
fi
export PATH="$WORKDIR/.uv:$WORKDIR/.uv/bin:$PATH"
if ! command -v uv > /dev/null; then
  say "installing uv into $WORKDIR/.uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WORKDIR/.uv" INSTALLER_NO_MODIFY_PATH=1 sh
fi
stamp="$TORCH_CUDA $(sha256sum requirements.txt | cut -c1-16)"
if [[ ! -x $PY || $(cat "$VENV/.installed" 2>/dev/null) != "$stamp" ]]; then
  say "building Python 3.11 env in $VENV (torch 2.10.0+$TORCH_CUDA; a few minutes)"
  uv venv --python 3.11 --allow-existing "$VENV"
  uv pip install --python "$PY" --index-strategy unsafe-best-match \
    --extra-index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
    "torch==2.10.0+$TORCH_CUDA" -r requirements.txt pytest
  echo "$stamp" > "$VENV/.installed"
fi
"$PY" - <<'EOF'
import torch
assert hasattr(torch.optim, "Muon"), "this torch has no torch.optim.Muon"
assert torch.cuda.is_available(), "torch cannot see CUDA (driver too old for this wheel?)"
for i in range(torch.cuda.device_count()):
    # Run real kernels: a wheel can serve a GPU through a compatible arch (sm_86 code on
    # sm_89), so the arch list alone is not the test.
    name, (major, minor) = torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)
    try:
        x = torch.randn(512, 512, device=f"cuda:{i}")
        product = (x @ x.T).cpu()
        assert torch.isfinite(torch.nn.functional.gelu(x).softmax(-1).sum()).item()
    except RuntimeError as exc:
        raise SystemExit(f"GPU {i} ({name}, sm_{major}{minor}) cannot run torch {torch.__version__} "
                         f"(arches {torch.cuda.get_arch_list()}): {exc}. Set TORCH_CUDA.")
    torch.testing.assert_close(product, x.cpu() @ x.cpu().T, rtol=1e-3, atol=1e-2)
    print(f"GPU {i}: {name} (sm_{major}{minor}) OK", flush=True)
print(f"torch {torch.__version__}: {torch.cuda.device_count()} GPU(s) OK")
EOF

# ---------------------------------------------------------------- data
say "fetching the ace_a_nme data into $REPO/datasets/ts"
(cd py && "$PY" - <<'EOF'
from common import ts_systems
from configs import ace_a_nme
system = ts_systems.resolve(ace_a_nme.get_config(0))
print(system.trajectory_name, ts_systems.ensure_data(system)["source"], flush=True)
EOF
)

# ---------------------------------------------------------------- tests
# Only the GPU path the runs use: CUDA-graph capture must match eager training exactly.
if [[ $RUN_TESTS == 1 && ! -f $WORKDIR/tests_passed_$commit ]]; then
  say "running the CUDA-graph tests on GPU 0 (about a minute)"
  CUDA_VISIBLE_DEVICES=0 "$PY" -m pytest py/tests/test_cuda_graphs.py -q -rfs -p no:cacheprovider \
    > "$LOGS/tests_cuda.log" 2>&1 || { tail -n 40 "$LOGS/tests_cuda.log"; die "CUDA tests failed; see $LOGS/tests_cuda.log"; }
  tail -n 1 "$LOGS/tests_cuda.log"
  touch "$WORKDIR/tests_passed_$commit"
fi

# ---------------------------------------------------------------- ablation
args=(run --root "$ROOT" --data_dir "$REPO/datasets/ts" --recipe "$RECIPE")
(( TEACHER_STEPS > 0 )) && args+=(--teacher_steps "$TEACHER_STEPS")
(( DISTILL_STEPS > 0 )) && args+=(--distill_steps "$DISTILL_STEPS")
(( GPUS > 0 )) && args+=(--n_gpus "$GPUS")
(( RUNS_PER_GPU > 0 )) && args+=(--runs_per_gpu "$RUNS_PER_GPU")
say "launching the $RECIPE-recipe ablation: teacher, four runs, evaluation (progress tables follow)"
"$PY" "${ORCHESTRATOR[@]}" "${args[@]}"
say "===== done: $ROOT/report.md ====="
