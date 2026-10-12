"""PyTorch port of common/updates.py (train step + EMA update)."""

from typing import Callable, Dict, Tuple

import torch

from . import distributed
from .cuda_graphs import CudaGraphLoss


def _all_finite(tensors, device):
    flags = [torch.isfinite(x).all() for x in tensors if x is not None]
    return torch.stack(flags).all() if flags else torch.tensor(True, device=device)


def setup_train_step(cfg) -> Callable:
    clip = cfg.optimization.clip
    use_graphs = getattr(getattr(cfg, "training", None), "cuda_graphs", False)
    graph_loss = CudaGraphLoss() if use_graphs else None

    def train_step(
        state,
        loss_fn: Callable,
        teacher_diag: Dict[str, torch.Tensor],
        teacher_offdiag: Dict[str, torch.Tensor],
        loss_args: Tuple,
        generator: torch.Generator,
    ):
        net = state.net
        optimizer = state.optimizer
        x0, x1, label, s, t, u, h = loss_args

        optimizer.zero_grad(set_to_none=True)
        density_kwargs = {}
        if getattr(getattr(cfg, "network", None), "predict_divergence", False):
            # Separate from the velocity teacher; fixed through forward/backward.
            density_kwargs["density_ema_params"] = state.ema_teacher(
                getattr(cfg.training, "density_ema_fac", None)
            )
        if graph_loss is None:
            loss_value, metrics = loss_fn(
                teacher_diag, teacher_offdiag, x0, x1, label, s, t, u, h,
                generator=generator, **density_kwargs
            )
            loss_value.backward()
        else:
            loss_value, metrics = graph_loss(
                state, loss_fn, teacher_diag, teacher_offdiag, loss_args,
                generator, **density_kwargs
            )

        grads = [p.grad for p in net.parameters() if p.grad is not None]
        loss_finite = torch.isfinite(loss_value).all()
        grads_finite = _all_finite(grads, loss_value.device)
        loss_finite, grads_finite = distributed.finite_consensus(loss_finite, grads_finite)
        all_finite = loss_finite and grads_finite

        # update learning rate per schedule
        lr = state.schedule(state.step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_multiplier", 1.0)

        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip)

        # Match the JAX reference (updates.py: jax.lax.cond(all_finite, ...)):
        # on a non-finite step neither the optimizer update nor the step counter
        # advances, so the LR schedule / EMA timing stay frozen until recovery.
        if all_finite:
            optimizer.step()
            state.step += 1
        state.last_update_succeeded = all_finite

        metrics = dict(metrics)
        metrics["train/loss_finite"] = bool(loss_finite)
        metrics["train/grads_finite"] = bool(grads_finite)
        metrics["train/nan_detected"] = (not all_finite)
        metrics["train/grad_norm"] = grad_norm.detach()
        metrics["train/lr"] = float(lr)
        for pg in optimizer.param_groups:
            if "group_name" in pg:
                metrics[f"train/lr_{pg['group_name']}"] = float(pg["lr"])
        return loss_value.detach(), metrics

    train_step.cuda_graph = graph_loss
    return train_step


def setup_ema_update(cfg) -> Callable:
    def update_ema_params(state):
        if getattr(state, "last_update_succeeded", True):
            state.update_ema()
        return state

    return update_ema_params
