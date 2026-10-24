"""Cycle-loss ablation on ace_a_nme: one velocity teacher, four SCALLOP runs, one report.

The runs differ only in the off-diagonal time pairs and the cycle loss. The code
jumps from s to t, so the "t < s" triangle of notes.tex (a jump from t to s) is
the code's ``forward`` order, s <= t:

    run1_square         (s, t) uniform on [0, 1]^2
    run2_forward        (s, t) uniform on the triangle s <= t (the recipe default)
    run3_square_cycle   run1 plus --cycle_loss
    run4_forward_cycle  run2 plus --cycle_loss

Each stage is the README command (velocity teacher, then ``--mode dlogpdt`` from
that teacher) with only these flags added, so all four share the teacher, the
seed and every recipe hyperparameter. Each run is evaluated with recipe v1 and
eval_round_trip.py. Stages resume from their latest checkpoint after a crash, a
stall or a rerun of this script. A short smoke pass at full batch size first
checks the pipeline and picks how many runs fit on one GPU.

    python py/launchers/cycle_ablation.py run --root outputs/cycle_ablation
    python py/launchers/cycle_ablation.py status --root outputs/cycle_ablation
    python py/launchers/cycle_ablation.py report --root outputs/cycle_ablation
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from typing import Callable, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

CFG_PATH = "configs.ace_a_nme"
EVAL_STEPS = 4  # recipe v1 flow-map jumps for ace_a_nme
TEACHER_STEPS, DISTILL_STEPS = 400_000, 100_000
RUNS = (
    ("run1_square", "square", False),
    ("run2_forward", "forward", False),
    ("run3_square_cycle", "square", True),
    ("run4_forward_cycle", "forward", True),
)
# README / recipe values every distillation run must have resolved to.
RECIPE = {
    ("network", "divergence_mode"): "dlogpdt",
    ("network", "div_loss_weighting_type"): "linear",
    ("network", "vectorized_divergence"): True,
    ("training", "loss_type"): "lsd",
    ("training", "stopgrad_type"): "stopped",
    ("training", "seed"): 42,
    ("training", "diag_teacher_source"): "external",
    ("training", "offdiag_teacher_source"): "external",
    ("optimization", "bs"): 1024,
    ("optimization", "diag_fraction"): .5,
    ("optimization", "optimizer"): "muon_adamw",
    ("optimization", "learning_rate"): 1e-4,
    ("problem", "com_augmentation"): True,
}
# arXiv:2606.29110 Table 1, ALA-2 at NFE 4: mean, std over 3 seeds; 2e5 SNIS samples.
PAPER = {
    "SCALLOP": {"ess": (.088, .001), "energy-w2": (.555, .116), "torus-w2": (.310, .027)},
    "F2D2": {"ess": (.066, .001), "energy-w2": (.958, .114), "torus-w2": (.383, .022)},
    "FALCON": {"ess": (.067, .013), "energy-w2": (.225, .104), "torus-w2": (.402, .021)},
    "FALCON-A": {"ess": (.097, .007), "energy-w2": (.512, .038), "torus-w2": (.180, .005)},
}
SMOKE_EVAL = ["--n_generate_samples", "20000", "--n_eval_samples", "1000", "--posthoc_mala_steps", "128"]


def log(message):
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def write_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(value, fh, indent=2)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "-"
    seconds = int(seconds)
    if seconds >= 86400:
        return f"{seconds // 86400}d{seconds % 86400 // 3600:02d}h"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def tail(path, n=20):
    try:
        with open(path, "rb") as fh:
            fh.seek(max(0, os.path.getsize(path) - 65536))
            text = fh.read().decode(errors="replace").replace("\r", "\n")
    except FileNotFoundError:
        return []
    return [line for line in text.splitlines() if line.strip()][-n:]


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def output_name(out_dir):
    cfg = read_json(os.path.join(out_dir, "config.json"))
    return None if cfg is None else cfg["logging"]["output_name"]


def checkpoint_step(path):
    import torch
    return int(torch.load(path, map_location="cpu", weights_only=False).get("step", 0))


def latest_checkpoint(out_dir):
    """(path, step) of the furthest saved training state, or (None, 0)."""
    name = output_name(out_dir)
    best = (None, 0)
    for suffix in ("", "_signal"):
        path = os.path.join(out_dir, f"{name}{suffix}.pt") if name else None
        if path and os.path.exists(path):
            step = checkpoint_step(path)
            if step > best[1]:
                best = (path, step)
    return best


def final_checkpoint(out_dir, total):
    """The run's final checkpoint if it reached ``total`` optimizer steps."""
    name = output_name(out_dir)
    path = os.path.join(out_dir, f"{name}.pt") if name else None
    return path if path and os.path.exists(path) and checkpoint_step(path) == total else None


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Job:
    name: str
    kind: str  # train | eval
    out_dir: str
    log_path: str
    argv: Callable[[Optional[str]], List[str]]  # resume checkpoint -> command
    is_done: Callable[[], bool]
    total: int = 0
    max_attempts: int = 3
    status: str = "pending"
    gpu: Optional[int] = None
    proc: Optional[subprocess.Popen] = None
    attempts: int = 0
    resumes: list = dataclasses.field(default_factory=list)
    oom: bool = False
    started: float = 0.0
    step: int = 0
    loss: Optional[float] = None
    cycle: Optional[float] = None
    ess: Optional[float] = None
    offset: int = 0
    history: deque = dataclasses.field(default_factory=lambda: deque(maxlen=512))
    warned: bool = False

    def activity(self):
        paths = (self.log_path, os.path.join(self.out_dir, "metrics.jsonl"))
        return max((os.path.getmtime(p) for p in paths if os.path.exists(p)), default=self.started)

    def read_metrics(self):
        path = os.path.join(self.out_dir, "metrics.jsonl")
        if self.kind != "train" or not os.path.exists(path):
            return
        with open(path, "rb") as fh:
            fh.seek(self.offset)
            data = fh.read()
        complete = data[:data.rfind(b"\n") + 1]
        self.offset += len(complete)
        for line in complete.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.step = int(record.get("step", self.step))
            self.loss = record.get("train/loss", self.loss)
            self.cycle = record.get("lsd/cycle_v_loss", self.cycle)
            for key, value in record.items():
                if key.endswith("resampled/effective-sample-size") and "/mean/" not in key:
                    self.ess = value
        if not self.history or self.history[-1][1] != self.step:
            self.history.append((time.time(), self.step))

    def rate(self, window=1800.0):
        """Optimizer steps per second over the last ``window`` seconds."""
        recent = [h for h in self.history if time.time() - h[0] <= window]
        if len(recent) < 2 or recent[-1][0] - recent[0][0] < 1:
            return None
        return (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])


class Scheduler:
    """Runs jobs on GPU slots, resumes failures, restarts stalls, prints progress."""

    def __init__(self, opts, root, title):
        self.opts, self.root, self.title = opts, root, title
        self.started = time.time()
        self.last_report = 0.0
        self.stop = None

    def _gpus(self):
        return list(range(self.opts.n_gpus)) if self.opts.device == "cuda" else [None]

    def _launch(self, job, gpu, threads):
        resume = None
        if job.kind == "train":
            resume, step = latest_checkpoint(job.out_dir) if os.path.exists(job.out_dir) else (None, 0)
            if resume:
                job.resumes.append(step)
                job.step = step
        argv = job.argv(resume)
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(threads),
                   CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
        with open(job.log_path, "a") as fh:
            fh.write(f"\n===== attempt {job.attempts + 1} {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
                     f"gpu={gpu} resume={resume}\n$ {' '.join(argv)}\n")
            job.proc = subprocess.Popen(argv, cwd=_REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        job.attempts += 1
        job.status, job.gpu, job.started, job.warned = "running", gpu, time.time(), False
        job.history.clear()
        log(f"{self.title}: started {job.name} on {'cpu' if gpu is None else f'GPU {gpu}'}"
            + (f", resuming from step {job.step}" if resume else ""))

    def _terminate(self, job, grace=180):
        try:
            os.killpg(job.proc.pid, signal.SIGTERM)  # learn.py saves *_signal.pt on SIGTERM
            job.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(job.proc.pid, signal.SIGKILL)
            job.proc.wait()
        except ProcessLookupError:
            pass

    def _finish(self, job, code):
        job.proc = None
        if code == 0 and job.is_done():
            job.status = "done"
            log(f"{self.title}: {job.name} finished")
            return
        lines = tail(job.log_path, 40)
        job.oom = any("out of memory" in line.lower() for line in lines)
        log(f"{self.title}: {job.name} exited with code {code}"
            + (" (CUDA out of memory)" if job.oom else "") + "; last log lines:\n    "
            + "\n    ".join(lines[-15:]))
        job.status = "pending" if job.attempts < job.max_attempts else "failed"
        if job.status == "pending":
            log(f"{self.title}: will retry {job.name} (attempt {job.attempts + 1}/{job.max_attempts})")

    def run(self, jobs, per_gpu):
        self.stop = None
        handlers = {sig: signal.signal(sig, lambda s, f: setattr(self, "stop", s))
                    for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            return self._loop(jobs, per_gpu)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)

    def _loop(self, jobs, per_gpu):
        gpus = self._gpus()
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        threads = max(1, cpus // max(1, min(len(jobs), per_gpu * len(gpus))))
        for job in jobs:
            if job.status == "pending" and job.is_done():
                job.status = "done"
                log(f"{self.title}: {job.name} already complete, skipping")
        while any(job.status in ("pending", "running") for job in jobs):
            if self.stop is not None:
                log(f"{self.title}: received signal {self.stop}; stopping jobs (training saves a checkpoint)")
                for job in jobs:
                    if job.status == "running":
                        self._terminate(job)
                raise SystemExit(128 + self.stop)
            for job in jobs:
                if job.status == "running":
                    job.read_metrics()
                    code = job.proc.poll()
                    if code is not None:
                        self._finish(job, code)
                    else:
                        self._check_stall(job)
            for job in jobs:
                if job.status != "pending":
                    continue
                load = {g: sum(j.status == "running" and j.gpu == g for j in jobs) for g in gpus}
                gpu = min(gpus, key=lambda g: load[g])
                if load[gpu] >= per_gpu:
                    break
                self._launch(job, gpu, threads)
            if time.time() - self.last_report >= (60 if time.time() - self.started < 900 else self.opts.progress_every):
                try:
                    self.report(jobs)
                except Exception as exc:  # noqa: BLE001 - progress output must never stop training
                    self.last_report = time.time()
                    log(f"{self.title}: progress report failed ({type(exc).__name__}: {exc})")
            time.sleep(self.opts.poll)
        self.report(jobs)
        return [job for job in jobs if job.status == "failed"]

    def _check_stall(self, job):
        quiet = time.time() - job.activity()
        warn = self.opts.stall_warn if job.kind == "train" else 3 * self.opts.stall_warn
        if quiet > warn and not job.warned:
            job.warned = True
            log(f"{self.title}: WARNING {job.name} has written nothing for {duration(quiet)}"
                + ("" if job.kind == "train" else " (MALA prints nothing; this can be normal)"))
        if job.kind == "train" and self.opts.stall_restart and quiet > self.opts.stall_restart:
            log(f"{self.title}: {job.name} stalled for {duration(quiet)}; restarting from its latest checkpoint")
            self._terminate(job)
            job.proc, job.status = None, ("pending" if job.attempts < job.max_attempts else "failed")

    def report(self, jobs):
        self.last_report = time.time()
        lines = [f"===== {self.title} | {datetime.datetime.now():%Y-%m-%d %H:%M:%S} | "
                 f"elapsed {duration(time.time() - self.started)} ====="]
        header = f"{'job':<30}{'state':<9}{'gpu':>4} {'step':>15} {'%':>6} {'it/s':>6} {'ETA':>8} " \
                 f"{'loss':>10} {'cycle_v':>9} {'val ESS':>8} {'quiet':>7}"
        lines.append(header)
        for job in jobs:
            rate = job.rate()
            pct = 100 * job.step / job.total if job.total else None
            eta = (job.total - job.step) / rate if rate and job.total else None
            quiet = time.time() - job.activity() if job.status == "running" else None

            def num(v, fmt):
                return "-" if v is None else format(v, fmt)
            lines.append(
                f"{job.name:<30}{job.status:<9}{num(job.gpu, 'd'):>4} "
                f"{(f'{job.step}/{job.total}' if job.total else '-'):>15} {num(pct, '.1f'):>6} "
                f"{num(rate, '.2f'):>6} {duration(eta):>8} {num(job.loss, '.4g'):>10} "
                f"{num(job.cycle, '.3g'):>9} {num(job.ess, '.4f'):>8} {duration(quiet):>7}")
            if job.kind == "eval" and job.status == "running":
                last = tail(job.log_path, 1)
                lines.append(f"    last output: {last[-1][:110] if last else '-'}")
        lines += gpu_lines()
        text = "\n".join(lines)
        print(text, flush=True)
        with open(os.path.join(self.root, "status.txt"), "w") as fh:
            fh.write(text + "\n")


def gpu_lines():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    rows = [r.split(", ") for r in out.stdout.strip().splitlines() if r.count(",") == 3]
    return [f"GPU {i}: util {u}% | memory {int(m) / 1024:.1f}/{int(t) / 1024:.1f} GB" for i, u, m, t in rows]


def stop_orphans(root, grace=180):
    """SIGTERM leftover stage processes of an earlier orchestrator so a relaunch never duplicates them."""
    pids = []
    for entry in os.listdir("/proc") if os.path.isdir("/proc") else []:
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                argv = fh.read().decode(errors="replace").split("\0")
        except (OSError, ValueError):
            continue
        if entry.isdigit() and int(entry) != os.getpid() and root in " ".join(argv) and any(
                a.endswith(("learn.py", "eval_peptide_ts.py", "eval_round_trip.py")) for a in argv):
            pids.append(int(entry))
    for pid in pids:
        log(f"stopping leftover stage process {pid} from an earlier launch (it saves a checkpoint)")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace
    while pids and time.time() < deadline:
        pids = [pid for pid in pids if os.path.exists(f"/proc/{pid}")]
        time.sleep(2)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        return sum(line.startswith("GPU ") for line in out.stdout.splitlines())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    def __init__(self, opts, root, smoke):
        self.opts, self.root, self.smoke = opts, root, smoke
        self.teacher_steps = opts.smoke_steps if smoke else TEACHER_STEPS
        self.distill_steps = opts.smoke_steps if smoke else DISTILL_STEPS
        os.makedirs(root, exist_ok=True)
        self.scheduler = Scheduler(opts, root, "smoke" if smoke else "ablation")

    def _learn(self, out_dir, extra):
        def argv(resume):
            cmd = [sys.executable, "py/launchers/learn.py", "--cfg_path", CFG_PATH,
                   "--ts_data_dir", self.opts.data_dir, "--output_folder", out_dir] + extra
            if self.smoke:
                freq = str(max(1, self.opts.smoke_steps // 2))
                cmd += ["--total_steps", str(self.opts.smoke_steps), "--visual_freq", freq, "--save_freq", freq]
            if self.opts.device == "cpu":
                cmd += ["--device", "cpu"]
            if resume:
                cmd += ["--network_load_path", resume, "--resume_optimizer"]
            return cmd
        return argv

    def teacher_job(self):
        out_dir = os.path.join(self.root, "teacher")
        return Job("teacher", "train", out_dir, os.path.join(self.root, "logs", "teacher.log"),
                   self._learn(out_dir, []), lambda: final_checkpoint(out_dir, self.teacher_steps) is not None,
                   total=self.teacher_steps, max_attempts=self.opts.max_attempts)

    def run_jobs(self, teacher, max_attempts):
        jobs = []
        for name, order, cycle in RUNS:
            out_dir = os.path.join(self.root, "runs", name)
            extra = ["--mode", "dlogpdt", "--network_load_path", teacher, "--teacher_load_path", teacher,
                     "--offdiag_time_order", order] + (["--cycle_loss"] if cycle else [])
            base = self._learn(out_dir, extra)

            def argv(resume, base=base, teacher=teacher):
                cmd = base(None)
                if resume:  # the student restarts from its own state; the teacher stays fixed
                    cmd[cmd.index("--network_load_path") + 1] = resume
                    cmd.append("--resume_optimizer")
                return cmd
            jobs.append(Job(name, "train", out_dir, os.path.join(self.root, "logs", f"{name}.log"), argv,
                            lambda d=out_dir: final_checkpoint(d, self.distill_steps) is not None,
                            total=self.distill_steps, max_attempts=max_attempts))
        return jobs

    def eval_jobs(self):
        jobs = []
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        workers = str(max(1, (cpus - 1) // len(RUNS)))
        for name, _, _ in RUNS:
            ckpt = final_checkpoint(os.path.join(self.root, "runs", name), self.distill_steps)
            out_dir = os.path.join(self.root, "eval", name)
            device = ["--device", self.opts.device]
            evaluate = [sys.executable, "py/launchers/eval_peptide_ts.py", "--cfg_path", CFG_PATH,
                        "--eval_recipe", "v1", "--ts_data_dir", self.opts.data_dir, "--out_dir", out_dir,
                        "--ckpt_path", ckpt, "--batch_size", "1000", "--n_workers", workers] + device
            round_trip = [sys.executable, "py/launchers/eval_round_trip.py", "--cfg_path", CFG_PATH,
                          "--ckpt_path", ckpt, "--out_dir", out_dir, "--ts_data_dir", self.opts.data_dir,
                          "--steps", str(EVAL_STEPS)] + device
            if self.smoke:
                evaluate += SMOKE_EVAL
                round_trip += ["--n_samples", "2000"]
            done_recipe = lambda d=out_dir, c=ckpt: (read_json(os.path.join(d, "ts_metrics.json"), {})
                                                    .get("provenance", {}).get("checkpoint_sha256") == sha256(c))
            done_trip = lambda d=out_dir, c=ckpt: (read_json(os.path.join(d, "round_trip.json"), {})
                                                  .get("checkpoint_sha256") == sha256(c))
            logs = os.path.join(self.root, "logs")
            jobs.append(Job(f"{name}:eval", "eval", out_dir, os.path.join(logs, f"{name}.eval.log"),
                            lambda r, a=evaluate: a, done_recipe, max_attempts=2))
            jobs.append(Job(f"{name}:round_trip", "eval", out_dir, os.path.join(logs, f"{name}.round_trip.log"),
                            lambda r, a=round_trip: a, done_trip, max_attempts=2))
        return jobs

    def execute(self, per_gpu):
        manifest_path = os.path.join(self.root, "manifest.json")
        manifest = read_json(manifest_path, {})
        manifest.setdefault("created", datetime.datetime.now().isoformat(timespec="seconds"))
        manifest.update(smoke=self.smoke, commit=git_commit(), device=self.opts.device,
                        n_gpus=self.opts.n_gpus, runs_per_gpu=per_gpu,
                        teacher_steps=self.teacher_steps, distill_steps=self.distill_steps)
        write_json(manifest_path, manifest)

        teacher = self.teacher_job()
        if self.scheduler.run([teacher], 1):
            raise RuntimeError("The velocity teacher failed; see logs/teacher.log")
        ckpt = final_checkpoint(teacher.out_dir, self.teacher_steps)
        manifest["teacher"] = {"checkpoint": ckpt, "sha256": sha256(ckpt), "resumed_at": teacher.resumes}
        write_json(manifest_path, manifest)

        runs = self.run_jobs(ckpt, 1 if self.smoke else self.opts.max_attempts)
        failed = self.scheduler.run(runs, per_gpu)
        manifest["runs"] = {job.name: {"resumed_at": job.resumes, "attempts": job.attempts,
                                       "status": job.status, "oom": job.oom} for job in runs}
        write_json(manifest_path, manifest)
        if failed:
            return failed
        failed = self.scheduler.run(self.eval_jobs(), len(RUNS))
        report = write_report(self.root, self.smoke)
        print(report, flush=True)
        return failed


def git_commit():
    try:
        out = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", _REPO, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def load_run(root, name, total):
    out_dir = os.path.join(root, "runs", name)
    eval_dir = os.path.join(root, "eval", name)
    records = []
    try:
        with open(os.path.join(out_dir, "metrics.jsonl")) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        pass
    return {
        "config": read_json(os.path.join(out_dir, "config.json"), {}),
        "records": records,
        "checkpoint": final_checkpoint(out_dir, total) if os.path.exists(out_dir) else None,
        "ts": read_json(os.path.join(eval_dir, "ts_metrics.json")),
        "round_trip": read_json(os.path.join(eval_dir, "round_trip.json")),
    }


def verify(run, order, cycle, teacher, total):
    """{check: passed} for one run, read back from what the run actually wrote."""
    cfg = run["config"]

    def get(section, key):
        return cfg.get(section, {}).get(key)
    sha = sha256(run["checkpoint"]) if run["checkpoint"] else None
    return {
        f"offdiag_time_order is {order}": get("training", "offdiag_time_order") == order,
        f"cycle_loss is {'on' if cycle else 'off'}": bool(get("training", "cycle_loss")) == cycle,
        "cycle losses logged only if on": any("lsd/cycle_v_loss" in r for r in run["records"]) == cycle,
        "README recipe hyperparameters": all(get(*key) == value for key, value in RECIPE.items()),
        "shared teacher checkpoint": teacher is not None and get("teacher", "load_path") == teacher,
        f"reached {total} optimizer steps": run["checkpoint"] is not None,
        "recipe-v1 eval of final checkpoint": bool(run["ts"]) and sha is not None
        and run["ts"].get("provenance", {}).get("checkpoint_sha256") == sha,
        "round trip of final checkpoint": bool(run["round_trip"]) and sha is not None
        and run["round_trip"].get("checkpoint_sha256") == sha,
    }


def stage(run, split, name):
    try:
        steps = run["ts"]["splits"][split]["steps"][str(EVAL_STEPS)]
    except (KeyError, TypeError):
        return {}
    if name == "relaxed":
        relaxed = sorted((k for k in steps if k.startswith("relaxed-")), key=lambda k: int(k.split("-")[1]))
        name = relaxed[-1] if relaxed else name
    return steps.get(name, {}) if isinstance(steps.get(name), dict) else {}


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def write_report(root, smoke=False):
    manifest = read_json(os.path.join(root, "manifest.json"), {})
    smoke = smoke or manifest.get("smoke", False)
    total = manifest.get("distill_steps", DISTILL_STEPS)
    teacher = (manifest.get("teacher") or {}).get("checkpoint")
    runs = {name: load_run(root, name, total) for name, _, _ in RUNS}
    out, data = [], {"manifest": manifest, "runs": {}}
    title = "SMOKE TEST (tiny budgets; numbers are meaningless)" if smoke else "Cycle-loss ablation, ace_a_nme (ALA-2)"
    out += [f"# {title}", "",
            f"Commit `{manifest.get('commit')}`; teacher {manifest.get('teacher_steps')} steps, "
            f"distillation {total} steps per run; recipe-v1 evaluation at {EVAL_STEPS} steps; one seed per run.", ""]

    out += ["## Verification", "", "Read back from each run's config.json, metrics.jsonl, final checkpoint "
            "and evaluation provenance.", ""]
    checks = {name: verify(runs[name], order, cycle, teacher, total) for name, order, cycle in RUNS}
    for name, _, _ in RUNS:
        failed = [check for check, ok in checks[name].items() if not ok]
        out.append(f"- {name}: " + ("all checks passed (" + "; ".join(checks[name]) + ")" if not failed
                                    else "**FAILED: " + "; ".join(failed) + "**"))
    all_ok = all(all(c.values()) for c in checks.values())
    out += ["", "All checks passed." if all_ok else "**Some checks failed: do not compare the failing runs.**", ""]

    metric_cols = [("ESS (clipped)", "resampled", "effective-sample-size", max),
                   ("ESS (unclipped)", "resampled", "effective-sample-size-unfiltered", max),
                   ("E-W2", "resampled", "energy-w2", min), ("T-W2", "resampled", "torus-w2", min),
                   ("TICA-W2", "resampled", "tica-w2", min),
                   ("E-W2 MALA", "relaxed", "energy-w2", min), ("T-W2 MALA", "relaxed", "torus-w2", min),
                   ("E-W2 proposal", "proposal", "energy-w2", min), ("T-W2 proposal", "proposal", "torus-w2", min)]
    budget = next((r["ts"] for r in runs.values() if r["ts"]), {})
    for split in ("test", "val"):
        out += [f"## Sample quality, {split} split", "",
                f"ESS on the full pool of {budget.get('n_generate_samples')} proposals (clipped = top "
                f"{100 * budget.get('logw_quantile_filter', 0):g}% of log-weights dropped, the paper's convention); "
                f"W2 against {budget.get('n_eval_samples')} reference frames for SNIS-resampled samples, after "
                f"{budget.get('posthoc_mala_steps')} MALA steps, and for raw proposals.", "",
                "| run | " + " | ".join(c[0] for c in metric_cols) + " |", "|---|" + "---|" * len(metric_cols)]
        values = {name: [stage(runs[name], split, st).get(key) for _, st, key, _ in metric_cols] for name, _, _ in RUNS}
        best = []
        for i, (_, _, _, pick) in enumerate(metric_cols):
            finite = [v[i] for v in values.values() if isinstance(v[i], (int, float)) and math.isfinite(v[i])]
            best.append(pick(finite) if finite else None)
        for name, _, _ in RUNS:
            cells = [(f"**{fmt(v)}**" if v is not None and v == best[i] else fmt(v)) for i, v in enumerate(values[name])]
            out.append(f"| {name} | " + " | ".join(cells) + " |")
            data["runs"].setdefault(name, {})[split] = dict(zip((c[0] for c in metric_cols), values[name]))
        for method, ref in PAPER.items():
            cells = [f"{ref['ess'][0]} ± {ref['ess'][1]}", "-", f"{ref['energy-w2'][0]} ± {ref['energy-w2'][1]}",
                     f"{ref['torus-w2'][0]} ± {ref['torus-w2'][1]}"] + ["-"] * (len(metric_cols) - 4)
            out.append(f"| {method} (paper) | " + " | ".join(cells) + " |")
        out.append("")

    out += ["## Invertibility (learned inverse map, test split)", "",
            "Round trips use the same 4-step grid in both directions. The inverse-map NLL uses log q from the "
            "learned backward jumps, so it is meaningful only where those jumps are trained.", "",
            "| run | x0 round-trip RMS (base units) | abs(log q fwd - log q inv) (nats) | data round-trip RMS (nm) "
            "| inverse-map NLL (nats) |", "|---|---|---|---|---|"]
    for name, _, _ in RUNS:
        rt = runs[name]["round_trip"] or {}
        cells = [fmt((rt.get("model_x0_rms_error") or {}).get("mean")),
                 fmt((rt.get("model_logq_abs_gap_nats") or {}).get("mean")),
                 fmt((rt.get("data_rms_error_nm") or {}).get("mean")), fmt(rt.get("data_inverse_nll_nats"))]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
        data["runs"].setdefault(name, {})["round_trip"] = rt

    out += ["", "## Training", "", "| run | final train loss (mean of last 10 logs) | final cycle_v | final cycle_div "
            "| best val ESS during training | resumed at steps |", "|---|---|---|---|---|---|"]
    for name, _, _ in RUNS:
        records = runs[name]["records"]
        losses = [r["train/loss"] for r in records if "train/loss" in r][-10:]
        ess = [v for r in records for k, v in r.items()
               if k.endswith("resampled/effective-sample-size") and "/mean/" not in k]
        last = next((r for r in reversed(records) if "lsd/cycle_v_loss" in r), {})
        resumed = ((manifest.get("runs") or {}).get(name) or {}).get("resumed_at")
        out.append(f"| {name} | {fmt(sum(losses) / len(losses) if losses else None)} | "
                   f"{fmt(last.get('lsd/cycle_v_loss'))} | {fmt(last.get('lsd/cycle_div_loss'))} | "
                   f"{fmt(max(ess) if ess else None)} | {resumed or '-'} |")

    out += ["", "## Reading these numbers", "",
            "- One seed per run. The paper's seed std for SCALLOP (ESS ±0.001, E-W2 ±0.116, T-W2 ±0.027) "
            "is the scale of seed noise; gaps much smaller than that are not evidence.",
            "- The paper's rows are 3 seeds, 2e5 samples, 4M AdamW updates; this repo's recipe is 100k "
            "Muon distillation steps, 1e6 proposals and 10k resampled samples, so absolute values differ.",
            "- The paper does not name its split; transferable-samplers reports on test."]
    text = "\n".join(out) + "\n"
    with open(os.path.join(root, "report.md"), "w") as fh:
        fh.write(text)
    data["checks"] = checks
    write_json(os.path.join(root, "report.json"), data)
    return text


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def calibrate(opts, root):
    """Run the smoke pipeline, lowering runs per GPU until four runs train without OOM."""
    per_gpu = opts.runs_per_gpu or max(1, math.ceil(len(RUNS) / max(1, opts.n_gpus)))
    while True:
        smoke_root = os.path.join(root, "smoke")
        pipeline = Pipeline(opts, smoke_root, smoke=True)
        failed = pipeline.execute(per_gpu)
        if not failed:
            log(f"smoke: passed with {per_gpu} run(s) per GPU")
            write_json(os.path.join(smoke_root, "passed.json"), {"runs_per_gpu": per_gpu})
            return per_gpu
        if per_gpu > 1 and any(job.oom for job in failed):
            log(f"smoke: out of memory with {per_gpu} runs per GPU; retrying with {per_gpu - 1}")
            shutil.rmtree(os.path.join(smoke_root, "runs"), ignore_errors=True)
            per_gpu -= 1
            continue
        raise SystemExit(f"smoke test failed: {', '.join(job.name for job in failed)}; "
                         f"see {os.path.join(smoke_root, 'logs')}")


def command_run(opts):
    root = os.path.abspath(opts.root)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "orchestrator.pid"), "w") as fh:
        fh.write(str(os.getpid()))
    stop_orphans(root)
    if opts.device == "cuda":
        opts.n_gpus = opts.n_gpus or gpu_count()
        if opts.n_gpus < 1:
            raise SystemExit("No GPU found (nvidia-smi -L); pass --device cpu only for testing")
    else:
        opts.n_gpus = 1
    smoke = read_json(os.path.join(root, "smoke", "passed.json"))
    if not opts.skip_smoke and smoke is None:
        smoke = {"runs_per_gpu": calibrate(opts, root)}
    per_gpu = (opts.runs_per_gpu or read_json(os.path.join(root, "manifest.json"), {}).get("runs_per_gpu")
               or (smoke or {}).get("runs_per_gpu") or max(1, math.ceil(len(RUNS) / opts.n_gpus)))
    log(f"ablation: {len(RUNS)} runs, {opts.n_gpus} GPU(s), {per_gpu} run(s) per GPU")
    if opts.smoke_only:
        return
    failed = Pipeline(opts, root, smoke=False).execute(per_gpu)
    if failed:
        raise SystemExit(f"failed: {', '.join(job.name for job in failed)}; see {os.path.join(root, 'logs')}")
    log(f"ablation complete: {os.path.join(root, 'report.md')}")


def command_status(opts):
    root = os.path.abspath(opts.root)
    path = os.path.join(root, "status.txt")
    try:
        pid = int(open(os.path.join(root, "orchestrator.pid")).read())
        os.kill(pid, 0)
        alive = f"orchestrator running (pid {pid})"
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        alive = "orchestrator NOT running"
    if os.path.exists(path):
        print(open(path).read().rstrip())
        print(f"(snapshot {duration(time.time() - os.path.getmtime(path))} old; {alive})")
    else:
        print(f"no status yet; {alive}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=["run", "status", "report"])
    p.add_argument("--root", default=os.path.join(_REPO, "outputs", "cycle_ablation"))
    p.add_argument("--data_dir", default=os.path.join(_REPO, "datasets", "ts"))
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--n_gpus", type=int, default=0, help="GPUs to use, 0..n-1 (default: all)")
    p.add_argument("--runs_per_gpu", type=int, default=0,
                   help="concurrent runs per GPU (default: all four at once, lowered by the smoke test on OOM)")
    p.add_argument("--skip_smoke", action="store_true")
    p.add_argument("--smoke_only", action="store_true")
    p.add_argument("--smoke_steps", type=int, default=300)
    p.add_argument("--max_attempts", type=int, default=4, help="launches per training stage (resumes included)")
    p.add_argument("--progress_every", type=float, default=600, help="seconds between progress tables")
    p.add_argument("--poll", type=float, default=15)
    p.add_argument("--stall_warn", type=float, default=1800, help="warn after this many quiet seconds")
    p.add_argument("--stall_restart", type=float, default=5400,
                   help="restart a quiet training stage from its checkpoint after this many seconds (0: never)")
    return p.parse_args(argv)


def main(argv=None):
    opts = parse_args(argv)
    if opts.command == "run":
        command_run(opts)
    elif opts.command == "status":
        command_status(opts)
    else:
        print(write_report(os.path.abspath(opts.root)))


if __name__ == "__main__":
    main()
