"""torchrun setup and DDP around the complete differentiable loss.

The raw flow map stays in TrainState for functional teachers, EMA, evaluation,
and portable checkpoints. DDP sees one forward per loss, including all JVPs.
"""

from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def active():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if active() else 0


def world_size():
    return dist.get_world_size() if active() else 1


def is_main():
    return rank() == 0


def barrier():
    if active():
        dist.barrier()


def initialize(device, *, cuda_graphs=False):
    device = torch.device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
            if cuda_graphs:
                # Full-backward NCCL capture requires this before process-group init.
                os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
                os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            # Rank zero may spend substantial time on molecular evaluation.
            timeout=timedelta(hours=1),
            **({"device_id": device} if device.type == "cuda" else {}),
        )
    return device


def configure_batch(cfg):
    """Keep bs global in config/W&B; require an exactly representable local split."""
    size = world_size()
    bs = int(cfg.optimization.bs)
    if bs < size or bs % size:
        raise ValueError(f"Global batch size {bs} must be positive and divisible by {size}")
    if hasattr(cfg.optimization, "diag_fraction"):
        diag = max(1, int(bs * cfg.optimization.diag_fraction))
    else:
        diag = int(cfg.optimization.diag_bs)
    offdiag = bs - diag
    if diag < 0 or offdiag < 0:
        raise ValueError("Diagonal batch must lie within the global batch")
    if size > 1 and (diag % size or offdiag % size or diag < size):
        raise ValueError("Global diagonal and off-diagonal batches must divide evenly across ranks")
    cfg.training.ndevices = size
    cfg.optimization.local_bs = bs // size
    cfg.optimization.local_diag_bs = diag // size


class LossModule(torch.nn.Module):
    def __init__(self, net, loss_fn):
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)


def wrap_loss(net, loss_fn, device, *, cuda_graphs=False):
    if cuda_graphs and device.type != "cuda":
        raise ValueError("CUDA graphs require training on a CUDA device")
    if not active():
        return loss_fn

    def build():
        return DistributedDataParallel(
            LossModule(net, loss_fn),
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,  # Fixed DiT positional/frequency arrays.
            find_unused_parameters=not cuda_graphs,
            static_graph=cuda_graphs,
            gradient_as_bucket_view=True,
        )

    if not cuda_graphs:
        return build()
    if dist.get_backend() != "nccl":
        raise ValueError("Distributed CUDA graphs require the NCCL backend")
    if torch.cuda.nccl.version() < (2, 9, 6):
        raise ValueError("Distributed CUDA graphs require NCCL >= 2.9.6")
    if os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING") != "0":
        raise ValueError("Initialize with distributed.initialize(..., cuda_graphs=True) "
                         "before constructing graphed DDP")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        wrapped = build()
    torch.cuda.current_stream(device).wait_stream(stream)
    wrapped._cuda_graph_stream = stream
    return wrapped


def finite_consensus(loss_finite, grads_finite):
    flags = torch.stack((loss_finite, grads_finite)).to(torch.int32)
    if active():
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    return [bool(v) for v in flags.tolist()]


def mean_metrics(loss, metrics):
    """Called by every rank; values are means over equally sized local batches."""
    if not active():
        return loss, metrics
    keys = sorted(metrics)
    values = torch.stack([torch.as_tensor(v, device=loss.device, dtype=loss.dtype)
                          for v in [loss, *[metrics[k] for k in keys]]]).detach()
    dist.all_reduce(values)
    values /= world_size()
    return values[0], dict(zip(keys, values[1:]))
