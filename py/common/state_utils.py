"""Training state, stage learning-rate schedules, optimizers and EMA snapshots.

Native checkpoints identify the selected backbone; cross-backbone loads are rejected.
"""

import os
import math
from typing import Any, Callable, Dict, Optional

import torch

from . import flow_map as flow_map_mod


def _model_tensors(net):
    """Parameters and buffers, including DiT's non-persistent positional arrays."""
    return {**dict(net.named_parameters()), **dict(net.named_buffers())}


def _network_architecture(net):
    dit = getattr(net, "dit", None)
    if dit is None:
        return None
    return {
        "num_layers": dit.num_layers,
        "num_shared_layers": dit.num_shared_layers,
        "num_div_layers": dit.num_div_layers if dit.predict_divergence else 0,
    }


def _checkpoint_architecture(sd):
    """Native metadata, or the historical all-shared layout for old snapshots."""
    if sd.get("network_architecture") is not None:
        return {k: sd["network_architecture"][k]
                for k in ("num_layers", "num_shared_layers", "num_div_layers")}
    params = sd["params"]
    def depth(prefix):
        indices = {int(k[len(prefix):].split(".")[0]) for k in params if k.startswith(prefix)}
        return max(indices) + 1 if indices else 0
    layers = depth("dit.blocks.")
    if not layers:
        return None
    return {"num_layers": layers, "num_shared_layers": layers,
            "num_div_layers": depth("dit.div_blocks.")}


def _validate_checkpoint_version(sd, expected):
    if expected is None:
        return
    params = sd["params"]
    # Check both metadata and tensor names before the tolerant density-head restore.
    version = sd.get("model_version")
    input_width = params.get("dit.x_embedder.weight", torch.empty(0, 0)).shape[-1]
    if expected == "canonical_atom_dit_v2":
        signature = ("dit.blocks.0.attn.qkv.weight" in params
                     and "dit.blocks.0.attn.q.weight" not in params
                     and "dit.final_layer.linear.weight" in params and input_width == 3)
        # Only the complete canonical sweep variant is compatible (A2/A3).
        if (version == "aaaaaa_pretrain_v_canonical_v1"
                and (sd.get("network_architecture") or {}).get("aaaaaa_variant") == "canonical"):
            version = expected
    else:
        signature = ("dit.blocks.0.attn.q.weight" in params
                     and "dit.blocks.0.attn.qkv.weight" not in params
                     and "dit.final_linear.weight" in params and input_width == 6)
    if version not in (None, expected) or not signature:
        raise ValueError(f"Incompatible backbone checkpoint; expected {expected}, "
                         f"got {sd.get('model_version')!r} or mismatched tensor signature")


def _validate_checkpoint_architecture(net, sd):
    version = getattr(getattr(net, "dit", None), "architecture_version", None)
    _validate_checkpoint_version(sd, version)
    current, saved = _network_architecture(net), _checkpoint_architecture(sd)
    if current is None or saved is None:
        return
    if current["num_layers"] != saved["num_layers"]:
        raise ValueError(f"Checkpoint velocity depth {saved} does not match model {current}")
    # A six-block velocity model can seed either branching layout exactly. A
    # learned density branch cannot be reinterpreted at another branch point.
    has_density = any(k.startswith("dit.div_") for k in sd["params"])
    if net.dit.predict_divergence and has_density and saved != current:
        raise ValueError(
            f"Checkpoint density architecture {saved} does not match model {current}. "
            "Use the checkpoint's saved layout to resume/evaluate it, or initialize "
            "the new density architecture from a velocity-only checkpoint."
        )


def _inference_tensors(net, tensors):
    """Drop learned loss-weight heads that an inference model does not build.

    They only weight training losses; sampling and log q never read them.
    """
    if getattr(net, "use_weight", False):
        return tensors
    return {k: v for k, v in tensors.items() if not k.startswith("logvar_")}


def load_evaluation_model(cfg, path, device, *, use_ema=True, require_saved_ema=False,
                          ema_factor=None, velocity_only=False):
    """Read one native checkpoint and build an inference model without an optimizer.

    Generic evaluation fills missing EMA entries from raw weights, as training
    restore did. Recipe/saved-config evaluation requires complete saved EMA.
    Training's partial velocity-to-density restore remains separate.
    """
    if str(path).endswith(".pkl"):
        raise ValueError("Legacy Flax checkpoint loading is unsupported; use a matching-backbone native .pt checkpoint")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected_version = flow_map_mod.get_dit_class(cfg.network).architecture_version
    _validate_checkpoint_version(saved, expected_version)
    has_density = any(k.startswith("dit.div_") for k in saved["params"])
    if velocity_only and has_density:
        raise ValueError("Velocity evaluation requires a velocity checkpoint without a divergence head")
    if cfg.network.predict_divergence and not has_density:
        raise ValueError("Cannot evaluate density: checkpoint has no density head")
    layout = _checkpoint_architecture(saved)
    if layout is not None:
        cfg.network.dit_kwargs.update(layout)
    net = flow_map_mod.TorchFlowMap(cfg.network).to(device)
    _validate_checkpoint_architecture(net, saved)
    net.load_state_dict(_inference_tensors(net, saved["params"]), strict=True)
    selected_factor = None
    if use_ema:
        selected_factor = ema_factor if ema_factor is not None else getattr(cfg.logging, "visual_ema_factor", None)
        if selected_factor is None and cfg.training.ema_facs:
            selected_factor = cfg.training.ema_facs[0]
        ema = next((value for key, value in saved.get("ema_params", {}).items()
                    if float(key) == selected_factor), None)
        if ema is not None:
            ema = _inference_tensors(net, ema)
        tensors = _model_tensors(net)
        if require_saved_ema:
            if ema is None:
                raise ValueError(f"Evaluation requires complete saved EMA {selected_factor}; raw fallback is forbidden")
            expected = net.state_dict()
            if set(expected) - set(ema) or set(ema) - set(tensors):
                raise ValueError("Saved EMA keys do not match the evaluation model")
            for key, value in ema.items():
                if value.shape != tensors[key].shape:
                    raise ValueError(f"Saved EMA shape mismatch: {key}")
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite saved EMA weights")
        if ema is not None:
            with torch.no_grad():
                for key, value in ema.items():
                    if key in tensors and value.shape == tensors[key].shape:
                        tensors[key].copy_(value)
    net.eval()
    return net, int(saved.get("step", 0)), selected_factor


def setup_schedule(cfg) -> Callable[[int], float]:
    """LR at zero-based accepted update k; cosine reaches zero at total_steps."""
    opt = cfg.optimization
    lr = float(opt.learning_rate)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if opt.schedule_type == "cosine":
        total = opt.total_steps
        warmup_fraction = float(getattr(opt, "warmup_fraction", 0.0))
        if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
            raise ValueError("cosine total_steps must be a positive integer")
        if not math.isfinite(warmup_fraction) or not 0 <= warmup_fraction < 1:
            raise ValueError("warmup_fraction must be finite and in [0, 1)")
        warmup = int(total * warmup_fraction)
        def sched(step):
            if step < 0:
                raise ValueError("schedule step must be nonnegative")
            if warmup and step < warmup:
                return lr * step / warmup
            fraction = min((step - warmup) / (total - warmup), 1.0)
            return lr * 0.5 * (1.0 + math.cos(math.pi * fraction))
    elif opt.schedule_type == "sqrt":
        if opt.decay_steps <= 0:
            raise ValueError("decay_steps must be positive")
        warmup_fraction = float(getattr(opt, "warmup_fraction", 0.0))
        if not math.isfinite(warmup_fraction) or not 0 <= warmup_fraction < 1:
            raise ValueError("warmup_fraction must be finite and in [0, 1)")
        # EDM2-style linear ramp, then square-root annealing (the paper's schedule).
        warmup = int(opt.total_steps * warmup_fraction) if warmup_fraction else 0
        def sched(step):
            ramp = min(step / warmup, 1.0) if warmup else 1.0
            return lr * ramp / math.sqrt(max(step / opt.decay_steps, 1.0))
    elif opt.schedule_type == "constant":
        def sched(step):
            return lr
    else:
        raise ValueError(f"Unknown schedule type: {opt.schedule_type}")
    return sched


def setup_optimizer(cfg, net: torch.nn.Module):
    schedule = setup_schedule(cfg)
    weight_decay = float(getattr(cfg.optimization, "weight_decay", 0.0))
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    kind = getattr(cfg.optimization, "optimizer", "radam")
    if kind == "muon_adamw":
        from .optimizers import MuonAdamW
        opt = cfg.optimization
        return MuonAdamW(
            net, lr=opt.learning_rate,
            betas=(getattr(opt, "b1", .9), getattr(opt, "b2", .999)),
            eps=getattr(opt, "eps", 1e-8),
            momentum=getattr(opt, "muon_momentum", .95),
            ns_steps=getattr(opt, "muon_ns_steps", 5),
            muon_lr_multiplier=getattr(opt, "muon_lr_multiplier", 1.),
            adamw_lr_multiplier=getattr(opt, "adamw_lr_multiplier", 1.),
            weight_decay=weight_decay,
        ), schedule
    if kind == "adamw":
        tx = torch.optim.AdamW(
            net.parameters(),
            lr=cfg.optimization.learning_rate,
            betas=(getattr(cfg.optimization, "b1", 0.9), getattr(cfg.optimization, "b2", 0.999)),
            eps=getattr(cfg.optimization, "eps", 1e-8),
            weight_decay=weight_decay,
        )
        tx.optimizer_type = "adamw"
        return tx, schedule
    if kind != "radam":
        raise ValueError(f"Unknown optimizer: {kind}")
    tx = torch.optim.RAdam(
        net.parameters(),
        lr=cfg.optimization.learning_rate,
        betas=(getattr(cfg.optimization, "b1", 0.9), getattr(cfg.optimization, "b2", 0.999)),
        eps=getattr(cfg.optimization, "eps", 1e-8),
        weight_decay=weight_decay,
    )
    return tx, schedule


class TrainState:
    """Live network + optimizer + EMA copies."""

    def __init__(self, net, optimizer, schedule, ema_facs):
        self.net = net
        self.optimizer = optimizer
        self.schedule = schedule
        self.step = 0
        self._cuda_graph_generation = 0
        self.ema_facs = list(ema_facs)
        self.ema_params: Dict[float, Dict[str, torch.Tensor]] = {
            fac: {n: p.detach().clone() for n, p in _model_tensors(net).items()}
            for fac in self.ema_facs
        }
        # Audit N3: best-metric trackers live here (not only in a module global in
        # logging.py) so they survive a restart and cannot be reset to +-inf, which
        # would make the first evaluation of a resumed run unconditionally "best"
        # and overwrite {output_name}_best_ess.pt with a worse checkpoint.
        self.best_metrics: Dict[str, float] = {}

    @torch.no_grad()
    def update_ema(self):
        for fac in self.ema_facs:
            ema = self.ema_params[fac]
            for n, p in self.net.named_parameters():
                ema[n].mul_(fac).add_(p.detach(), alpha=(1.0 - fac))
            # DiT buffers are fixed; snapshot them too rather than borrowing the
            # student state during functional EMA evaluation.
            for n, b in self.net.named_buffers():
                ema[n].copy_(b)

    def self_teacher(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach() for n, p in _model_tensors(self.net).items()}

    def ema_teacher(self, fac: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Select a detached EMA snapshot for velocity or density consistency.

        Density targets always use this snapshot independently of velocity
        selection. Its tensors must remain fixed until backward has finished.
        """
        if fac is None:
            if not self.ema_facs:
                raise ValueError("An EMA factor is required for density/EMA teacher evaluation")
            fac = self.ema_facs[0]
        if fac not in self.ema_facs or fac not in self.ema_params:
            raise ValueError(f"EMA factor {fac} is not updated by this run; configured: {self.ema_facs}")
        return {n: p.detach() for n, p in self.ema_params[fac].items()}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "params": self.net.state_dict(),
            "ema_params": self.ema_params,
            "opt_state": self.optimizer.state_dict(),
            "optimizer_type": getattr(self.optimizer, "optimizer_type", "radam"),
            "step": self.step,
            "best_metrics": dict(self.best_metrics),
            "network_architecture": _network_architecture(self.net),
            "model_version": getattr(getattr(self.net, "dit", None), "architecture_version", None),
        }

    def load_state_dict(self, sd: Dict[str, Any], load_optimizer: bool = True):
        _validate_checkpoint_architecture(self.net, sd)
        if load_optimizer and "opt_state" in sd:
            current_kind = getattr(self.optimizer, "optimizer_type", "radam")
            saved_kind = sd.get("optimizer_type", sd["opt_state"].get("optimizer_type", "radam"))
            if current_kind != saved_kind:
                raise ValueError("Changing optimizer type requires reset_optimizer=True")
        # Captures must not retain the EMA allocations replaced by this restore.
        self._cuda_graph_generation += 1
        # Tolerant restore (mirrors the JAX partial-restore): matching params are
        # loaded by name/shape; newly added params (e.g. a div head) keep their
        # freshly-initialised values.
        device = next(self.net.parameters()).device
        own = self.net.state_dict()
        filtered = {
            k: v for k, v in sd["params"].items()
            if k in own and tuple(own[k].shape) == tuple(v.shape)
        }
        missing = [k for k in own if k not in filtered]
        if missing:
            print(f"Checkpoint missing {len(missing)} params (kept at init), e.g. {missing[:3]}")
        self.net.load_state_dict(filtered, strict=False)

        # Seed EVERY missing/incompatible entry from the restored student, then
        # overlay saved history. A partially compatible factor must not retain
        # random pre-restore trunk weights or borrow checkpoint tensor storage.
        saved_ema = sd.get("ema_params", {})
        factors = set(self.ema_facs) | {float(f) for f in saved_ema}
        current = _model_tensors(self.net)
        self.ema_params = {
            fac: {n: v.detach().clone() for n, v in current.items()} for fac in factors
        }
        for fac, params in saved_ema.items():
            target = self.ema_params[float(fac)]
            for n, v in params.items():
                if n in target and target[n].shape == v.shape:
                    target[n] = v.detach().to(device=device, dtype=target[n].dtype).clone()
        if load_optimizer and "opt_state" in sd:
            try:
                self.optimizer.load_state_dict(sd["opt_state"])
            except Exception as exc:
                raise ValueError("Optimizer restore failed; explicitly reset_optimizer for a fresh stage") from exc
        self.step = sd.get("step", 0)
        # Audit N3: carry the best-metric trackers across the load.
        self.best_metrics = dict(sd.get("best_metrics") or {})


def setup_training_state(cfg, device):
    net = flow_map_mod.TorchFlowMap(cfg.network).to(device)
    optimizer, schedule = setup_optimizer(cfg, net)
    state = TrainState(net, optimizer, schedule, cfg.training.ema_facs)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {n_params}")

    load_path = getattr(cfg.network, "load_path", "")
    if load_path:
        print(f"Loading checkpoint from {load_path}.")
        load_checkpoint_into_state(cfg, state, load_path)
        if getattr(cfg.network, "reset_optimizer", False):
            print("Resetting optimizer state.")
            state.step = 0
            if getattr(cfg.network, "reset_best_metrics", False):
                state.best_metrics.clear()
    return state, net, schedule


def load_checkpoint_into_state(cfg, state: TrainState, path: str):
    if path.endswith(".pkl"):
        raise ValueError("Legacy Flax checkpoint loading is unsupported; use a matching-backbone native .pt checkpoint")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    state.load_state_dict(sd, load_optimizer=not getattr(cfg.network, "reset_optimizer", False))
    init_from_ema = getattr(cfg.network, "init_from_ema_factor", None)
    if init_from_ema is not None and init_from_ema in state.ema_params:
        print(f"Initializing params from EMA {init_from_ema}.")
        state.net.load_state_dict(state.ema_params[init_from_ema], strict=False)


def load_teacher_params(cfg, state: TrainState, device) -> Optional[Dict[str, torch.Tensor]]:
    teacher_cfg = getattr(cfg, "teacher", None)
    if teacher_cfg is None:
        return None
    teacher_path = getattr(teacher_cfg, "load_path", "")
    if not teacher_path:
        return None
    teacher_ema_fac = getattr(teacher_cfg, "ema_fac", None)

    print(f"Loading teacher checkpoint from {teacher_path}.")
    if teacher_path.endswith(".pkl"):
        raise ValueError("Legacy Flax teacher loading is unsupported; use a matching-backbone native .pt checkpoint")

    sd = torch.load(teacher_path, map_location=device, weights_only=False)
    _validate_checkpoint_architecture(state.net, sd)
    if teacher_ema_fac is not None and teacher_ema_fac in sd.get("ema_params", {}):
        params = sd["ema_params"][teacher_ema_fac]
    else:
        params = sd["params"]
    return {k: v.detach().to(device) for k, v in params.items()}


def save_state(state: TrainState, path: str):
    """Write the checkpoint atomically.

    EXPERIMENT_LOG 2026-07-29 records a truncated 1345-byte ``*_signal.pt`` where a
    ~100 MB checkpoint was expected: ``torch.save`` straight to the final path leaves
    a corrupt file if the process dies mid-write, and the SIGTERM handler in
    ``logging.register_signal_handlers`` swallows the exception so it looks clean.
    Write to a sibling temp file and ``os.replace`` (atomic within a filesystem), so a
    partial write can never destroy the previous good checkpoint.
    """
    tmp = f"{path}.tmp"
    try:
        torch.save(state.state_dict(), tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
