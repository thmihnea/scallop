"""CUDA capture of loss forward/backward, with eager optimizer/EMA updates.

Each run has fixed input layouts and a fixed loss/parameter-use pattern. Storage
replacement (including checkpoint restore) rebuilds the capture on every rank.
Normal in-place parameter/EMA updates and evaluation need no new capture.
"""
from contextvars import ContextVar

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import distributed


_checks = ContextVar('cuda_graph_validation', default=None)


def check_tensor(condition, message):
    """Raise eagerly, or retain a GPU flag to validate after captured backward."""
    checks = _checks.get()
    if checks is not None:
        checks.append((condition, message))
    elif not bool(condition):
        raise ValueError(message)


def _layout(value):
    if value is None:
        return None
    return (tuple(value.shape), tuple(value.stride()), value.dtype,
            value.device, value.requires_grad)


def _storage(tree):
    if tree is None:
        return None
    return tuple((name, value.data_ptr(), _layout(value)) for name, value in tree.items())


class _CapturedBackward:
    def __init__(self, state, loss_fn, teachers, args, generator, kwargs):
        self.state = state
        # Keep external allocations alive for the entire graph lifetime.
        self.loss_fn, self.teachers, self.kwargs = loss_fn, teachers, kwargs
        self.generator = generator
        self.inputs = tuple(None if x is None else x.detach().clone() for x in args)
        self.graph = torch.cuda.CUDAGraph()
        device = next(state.net.parameters()).device
        stream = getattr(loss_fn, "_cuda_graph_stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        # Warmup must not consume training probes, including on recapture.
        default_rng = torch.cuda.get_rng_state(device)
        custom_rng = generator.get_state() if generator is not None else None
        if generator is not None:
            self.graph.register_generator_state(generator)

        def calculate():
            state.net.zero_grad(set_to_none=True)
            checks = []
            token = _checks.set(checks)
            try:
                self.value, self.metrics = loss_fn(*teachers, *self.inputs,
                                                   generator=generator, **kwargs)
                self.value.backward()
            finally:
                _checks.reset(token)
            self.checks = checks

        try:
            # DDP's reducer/bucket setup must settle before full-backward capture.
            warmup = 11 if distributed.active() else 3
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    calculate()
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            state.net.zero_grad(set_to_none=True)
            with torch.cuda.graph(self.graph, stream=stream):
                calculate()
            self.grads = [(p, p.grad) for p in state.net.parameters()]
        finally:
            torch.cuda.set_rng_state(default_rng, device)
            if generator is not None:
                generator.set_state(custom_rng)

    def __call__(self, args):
        for target, value in zip(self.inputs, args):
            if target is not None:
                target.copy_(value)
        self.graph.replay()
        for parameter, grad in self.grads:
            parameter.grad = grad
        if self.checks:
            flags = torch.stack([flag for flag, _ in self.checks]).to(torch.int32)
            if distributed.active():
                # A bad density input on one rank must reject on all ranks.
                dist.all_reduce(flags, op=dist.ReduceOp.MIN)
            for valid, (_, message) in zip(flags.tolist(), self.checks):
                if not valid:
                    raise ValueError(message)
        return self.value, self.metrics


class CudaGraphLoss:
    """One lazily captured backward per training-step callable."""
    def __init__(self):
        self.capture = None
        self.key = None
        self.input_layouts = None
        self.capture_count = 0

    def __call__(self, state, loss_fn, teacher_diag, teacher_offdiag, args,
                 generator, **kwargs):
        device = next(state.net.parameters()).device
        if device.type != 'cuda':
            raise ValueError('CUDA graphs require training on a CUDA device')
        if generator is not None:
            generator_device = torch.device(generator.device)
            if generator_device.type == 'cuda' and generator_device.index is None:
                generator_device = torch.device('cuda', torch.cuda.current_device())
            if generator_device != device:
                raise ValueError('The training generator must use the model CUDA device')
        if isinstance(loss_fn, DistributedDataParallel) and not loss_fn.static_graph:
            raise ValueError('Construct DDP with distributed.wrap_loss(..., cuda_graphs=True)')
        layouts = tuple(_layout(x) for x in args)
        key = (id(state), id(loss_fn), id(generator), state.net.training,
               getattr(state, '_cuda_graph_generation', 0),
               _storage(dict(state.net.named_parameters())),
               _storage(dict(state.net.named_buffers())),
               _storage(teacher_diag), _storage(teacher_offdiag),
               _storage(kwargs.get('density_ema_params')))
        recapture = key != self.key
        wrong_layout = self.input_layouts is not None and layouts != self.input_layouts
        if distributed.active():
            flags = torch.tensor([recapture, wrong_layout], dtype=torch.int32, device=device)
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            recapture, wrong_layout = flags.tolist()
        if wrong_layout:
            raise ValueError('CUDA graphs require fixed input shapes, layouts and dtypes; '
                             'create a new training-step callable for a new configuration')
        if any(x is not None and (x.device != device or x.requires_grad) for x in args):
            raise ValueError('CUDA graph inputs must be detached tensors on the model device')
        if recapture:
            self.capture = None
            self.capture = _CapturedBackward(state, loss_fn, (teacher_diag, teacher_offdiag),
                                              args, generator, kwargs)
            self.key, self.input_layouts = key, layouts
            self.capture_count += 1
        return self.capture(args)
