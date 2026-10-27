"""PyTorch port of common/flow_map.py + common/network_utils.py (DiTFlowMap).

``TorchFlowMap`` merges the JAX ``FlowMap`` (which provided ``partial_t`` /
``partial_s`` via ``jax.jvp``) and ``DiTFlowMap`` (which wrapped the DiT network
with the EDM-style pre/post conditioning).

The ``forward`` method dispatches on a ``method`` keyword so the module can be
driven with alternative parameter sets through ``torch.func.functional_call``
(used to evaluate teacher parameters inside the losses).
"""

import math

import torch
import torch.nn as nn

from . import dit as dit_mod
from .dit_v2 import FlowMapAtomDiTV2


def _bcast(coef, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar / (B,) coefficient against tensor ``x``."""
    if not torch.is_tensor(coef):
        return coef
    coef = coef.to(dtype=x.dtype)
    while coef.ndim < x.ndim:
        coef = coef.unsqueeze(-1)
    return coef


class _MPLinear(nn.Module):
    """EDM2 ``MPConv``-equivalent linear (magnitude-preserving / forced weight norm).

    On every forward the weight is normalized to unit L2 norm over the fan-in and
    scaled by ``1/sqrt(fan_in)`` (matching ``edm2_net.normalize`` + ``MPConv``), so
    the output magnitude is bounded regardless of the raw weight's scale. This is
    what keeps the learned ``logvar`` from diverging -- a plain ``nn.Linear`` head
    let it blow up to ~1e17. The weight is zero-initialized so ``logvar == 0`` at
    init (the loss starts as the plain unweighted objective).
    """

    def __init__(self, in_features: int, eps: float = 1e-4):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, in_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.float()
        c = w.shape[1]
        # edm2_net.normalize(dim=fan_in): w / (eps + ||w||/sqrt(c)); then MPConv gain/sqrt(c).
        w = w / (self.eps + w.norm(dim=1, keepdim=True) / math.sqrt(c))
        w = w / math.sqrt(c)
        return (x.float() @ w.t()).to(x.dtype)


def get_dit_class(config):
    """Select a backbone explicitly; configs without a version keep the original."""
    version = getattr(config, "dit_version", dit_mod.FlowMapAtomDiT.architecture_version)
    if version == dit_mod.FlowMapAtomDiT.architecture_version:
        return dit_mod.FlowMapAtomDiT
    if version == FlowMapAtomDiTV2.architecture_version:
        return FlowMapAtomDiTV2
    raise ValueError(f"Unknown network.dit_version: {version!r}")


class TorchFlowMap(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dit_kwargs = dict(config.dit_kwargs)
        self.dit = get_dit_class(config)(
            n_particles=config.n_particles,
            n_dimension=config.n_dimension,
            predict_divergence=config.predict_divergence,
            vectorized_divergence=config.vectorized_divergence,
            com_aware=getattr(config, "com_aware", False),
            **dit_kwargs,
        )
        self.sigma_data = config.sigma_data
        self.use_weight = config.use_weight
        self.logvar_channels = config.logvar_channels
        self.use_bfloat16 = config.use_bfloat16
        # Mixed precision is INDEPENDENT of use_bfloat16 above: that one rounds the
        # coordinate input and leaves every matmul in fp32; this one puts the DiT
        # trunk's matmuls on bf16 tensor cores. "off" | "bf16_trunk".
        self.mixed_precision = str(getattr(config, "mixed_precision", "off"))
        if self.mixed_precision not in ("off", "bf16_trunk"):
            raise ValueError(
                f"Unknown network.mixed_precision: {self.mixed_precision!r} "
                "(expected 'off' or 'bf16_trunk')"
            )

        if self.use_weight:
            self.logvar_fourier_s = dit_mod.PositionalEmbedding(self.logvar_channels)
            self.logvar_fourier_t = dit_mod.PositionalEmbedding(self.logvar_channels)
            # magnitude-preserving (weight-normalized) head + zero init -> logvar==0 at
            # init and bounded thereafter (plain nn.Linear diverged to ~1e17).
            self.logvar_linear = _MPLinear(self.logvar_channels)
        # The cycle loss gets its own learned log-variance w_c(s, t) on the same features.
        self.cycle_logvar = bool(self.use_weight and getattr(config, "cycle_logvar", False))
        if self.cycle_logvar:
            self.logvar_cycle_linear = _MPLinear(self.logvar_channels)

    # ------------------------------------------------------------------ #
    # weight function
    # ------------------------------------------------------------------ #
    def _logvar_features(self, s, t):
        s = torch.as_tensor(s, dtype=torch.float32)
        t = torch.as_tensor(t, dtype=torch.float32)
        embed_s = self.logvar_fourier_s(s).reshape(-1, self.logvar_channels)
        embed_t = self.logvar_fourier_t(t).reshape(-1, self.logvar_channels)
        return (embed_s + embed_t) / math.sqrt(2.0)

    def calc_weight(self, s, t):
        if not self.use_weight:
            return 0.0  # log variance 0 => exactly the unweighted squared loss
        return self.logvar_linear(self._logvar_features(s, t)).reshape(-1, 1, 1)

    def calc_cycle_weight(self, s, t):
        """Learned log-variance of the cycle loss, or None when it shares calc_weight."""
        if not getattr(self, "cycle_logvar", False):
            return None
        return self.logvar_cycle_linear(self._logvar_features(s, t)).reshape(-1, 1, 1)

    # ------------------------------------------------------------------ #
    # core network evaluation
    # ------------------------------------------------------------------ #
    def process_input(self, s, t, x: torch.Tensor):
        s = s if torch.is_tensor(s) else torch.as_tensor(s, dtype=torch.float32, device=x.device)
        t = t if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.float32, device=x.device)
        s = s.to(torch.float32)
        t = t.to(torch.float32)
        if x.ndim == 2:
            x = x[None, ...]
            is_single = True
        elif x.ndim == 3:
            is_single = False
        else:
            raise ValueError(f"Unsupported x shape: {tuple(x.shape)}")

        batch_size = x.shape[0]
        s = s.reshape(-1, 1, 1).expand(batch_size, 1, 1)
        t = t.reshape(-1, 1, 1).expand(batch_size, 1, 1)
        return s, t, x, is_single

    def calc_phi(
        self,
        s,
        t,
        x,
        label=None,
        train: bool = True,
        calc_weight: bool = False,
        return_div: bool = False,
        return_velocity: bool = True,
    ):
        """Raw heads; unrequested branches are skipped before their evaluation.

        Default: velocity tensor. return_div=True: (velocity, density).
        With return_velocity=False: (None, density); calc_div is the tensor-only
        convenience API. Velocity-only models retain their legacy return shape.
        """
        if not return_velocity and (not return_div or not self.config.predict_divergence):
            raise ValueError("Density-only evaluation requires return_div=True and a density head")
        output = "both" if return_div and return_velocity else ("velocity" if return_velocity else "divergence")
        ss, ts, xs, is_single = self.process_input(s, t, x)

        c_out = self.sigma_data
        c_in = 1.0 / self.sigma_data

        # Match JAX: bf16 only truncates the network input; the Dense kernels are
        # float32 so every matmul promotes back to float32.
        xs_in = c_in * xs
        if self.use_bfloat16:
            xs_in = xs_in.to(torch.bfloat16).to(torch.float32)
        else:
            xs_in = xs_in.to(torch.float32)
        if self.mixed_precision == "bf16_trunk":
            # Scoped to exactly this call: the two lines below already cast the outputs
            # back to fp32, so everything downstream (div_hat, the LSD residuals, the
            # dlogp accumulator) stays fp32 by construction. The device type comes
            # from the tensor, not a hard "cuda", so the CPU-only tests exercise the
            # real path. dit.py pins its own fp32 islands (LayerNorm interior,
            # residual stream, conditioning, both output heads).
            with torch.amp.autocast(xs_in.device.type, dtype=torch.bfloat16):
                phi_raw, div_raw = self.dit(xs_in, ss, ts - ss, train=train, output=output)
        else:
            phi_raw, div_raw = self.dit(xs_in, ss, ts - ss, train=train, output=output)

        phi_st = c_out * phi_raw.to(torch.float32) if phi_raw is not None else None
        # Raw average-divergence head d, in units of 1/divergence_rescale.
        # Accumulated positive divergence D=(t-s)*d is formed once in _integrate.
        div_st = c_out * div_raw.to(torch.float32) if div_raw is not None else None

        if calc_weight:
            logvar = self.calc_weight(s, t)
            logvar0 = logvar if not torch.is_tensor(logvar) else logvar[0]
            if return_div and div_st is not None:
                if is_single:
                    return (None if phi_st is None else phi_st[0]), div_st[0], logvar0
                return phi_st, div_st, logvar
            if is_single:
                return phi_st[0], logvar0
            return phi_st, logvar

        if return_div and div_st is not None:
            if is_single:
                return (None if phi_st is None else phi_st[0]), div_st[0]
            return phi_st, div_st
        if is_single:
            return phi_st[0]
        return phi_st

    def calc_div(self, s, t, x, label=None, train=True, calc_weight=False):
        """Raw density only, skipping velocity-specific blocks and projection."""
        result = self.calc_phi(s, t, x, label=label, train=train,
                               calc_weight=calc_weight, return_div=True,
                               return_velocity=False)
        return (result[1], result[2]) if calc_weight else result[1]

    def calc_b(self, t, x, label=None, train=True, calc_weight=False, return_div=False):
        return self.calc_phi(
            t, t, x, label=label, train=train, calc_weight=calc_weight, return_div=return_div
        )

    def flow_call(
        self,
        s,
        t,
        x,
        label=None,
        train: bool = True,
        calc_weight: bool = False,
        return_X_and_phi: bool = False,
        return_div: bool = False,
    ):
        phi_rslt = self.calc_phi(
            s, t, x, label=label, train=train, calc_weight=calc_weight,
            return_div=return_div,
        )

        div_st = None
        if calc_weight:
            if return_div and isinstance(phi_rslt, tuple) and len(phi_rslt) == 3:
                phi_st, div_st, logvar = phi_rslt
            else:
                phi_st, logvar = phi_rslt
        else:
            if return_div and isinstance(phi_rslt, tuple) and len(phi_rslt) == 2:
                phi_st, div_st = phi_rslt
            else:
                phi_st = phi_rslt

        tms = _bcast(t - s, phi_st) if torch.is_tensor(t) or torch.is_tensor(s) else (t - s)
        Xst = x + tms * phi_st

        if calc_weight:
            if return_div and div_st is not None:
                return Xst, div_st, logvar
            return Xst, logvar
        if return_X_and_phi:
            if return_div and div_st is not None:
                return Xst, phi_st, div_st
            return Xst, phi_st
        if return_div and div_st is not None:
            return Xst, div_st
        return Xst

    # ------------------------------------------------------------------ #
    # derivatives via forward-mode AD (composable with reverse mode)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _jvp(fn, primal: torch.Tensor):
        # functorch forward-mode AD; composes cleanly with a later .backward()
        # (reverse-over-forward) unlike torch.autograd.forward_ad which trips on
        # softmax's in-place saved tensors.
        out, tangent = torch.func.jvp(fn, (primal,), (torch.ones_like(primal),))
        return out, tangent

    def partial_t(self, s, t, x, label=None, train=True, calc_weight=False, return_div=False):
        t = t if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.float32, device=x.device)
        t = t.to(torch.float32)

        def fn(tt):
            return self.flow_call(
                s, tt, x, label=label, train=train, calc_weight=calc_weight,
                return_X_and_phi=False, return_div=return_div,
            )

        return self._jvp(fn, t)

    def partial_s(self, s, t, x, label=None, train=True, calc_weight=False, return_div=False):
        s = s if torch.is_tensor(s) else torch.as_tensor(s, dtype=torch.float32, device=x.device)
        s = s.to(torch.float32)

        def fn(ss):
            return self.flow_call(
                ss, t, x, label=label, train=train, calc_weight=calc_weight,
                return_X_and_phi=False, return_div=return_div,
            )

        return self._jvp(fn, s)

    # ------------------------------------------------------------------ #
    # dispatcher (so functional_call can pick a method)
    # ------------------------------------------------------------------ #
    def forward(self, *args, method: str = "flow_call", **kwargs):
        if method == "flow_call":
            return self.flow_call(*args, **kwargs)
        if method == "calc_b":
            return self.calc_b(*args, **kwargs)
        if method == "calc_div":
            return self.calc_div(*args, **kwargs)
        if method == "calc_phi":
            return self.calc_phi(*args, **kwargs)
        if method == "partial_t":
            return self.partial_t(*args, **kwargs)
        if method == "partial_s":
            return self.partial_s(*args, **kwargs)
        if method == "calc_weight":
            return self.calc_weight(*args, **kwargs)
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------ #
    # base distribution / (un)scaling for sampling & likelihood
    # ------------------------------------------------------------------ #
    def base_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3 and x.shape[2] == 3, "x must be (batch, n_atoms, 3)"
        sigma = torch.as_tensor(self.config.rescale, dtype=x.dtype, device=x.device)
        if sigma.ndim == 0:
            sigma = torch.ones(x.shape[1:], dtype=x.dtype, device=x.device) * sigma
        dims = tuple(range(1, x.ndim))
        quad = torch.sum((x / sigma) ** 2, dim=dims)
        total_dim = float(x.shape[1] * x.shape[2])
        log_det = torch.sum(torch.log(sigma**2))
        log_norm = 0.5 * (total_dim * math.log(2 * math.pi) + log_det)
        unconstrained_log_prob = -0.5 * quad - log_norm
        if getattr(self.config, "com_aware", False):
            # com-aware: full 3N Gaussian base (COM is a modeled DOF); no mean-free
            # subspace correction. The COM density is handled in importance sampling
            # at the final importance-weight boundary (common.com_lift).
            return unconstrained_log_prob
        var_center_of_mass = torch.mean(sigma**2, dim=0)
        correction = 0.5 * (3 * math.log(2 * math.pi) + torch.sum(torch.log(var_center_of_mass)))
        return unconstrained_log_prob + correction

    @property
    def modeled_dimension(self):
        n = self.config.n_particles if getattr(self.config, "com_aware", False) else self.config.n_particles - 1
        return n * self.config.n_dimension

    def rescale_samples(self, x, logp=None):
        x = x / self.config.unscale_factor
        if logp is not None:
            logp = logp + self.modeled_dimension * math.log(self.config.unscale_factor)
            return x, logp
        return x

    def unscale_samples(self, x, logp=None):
        x = x * self.config.unscale_factor
        if logp is not None:
            logp = logp - self.modeled_dimension * math.log(self.config.unscale_factor)
            return x, logp
        return x

    @torch.no_grad()
    def _integrate(self, s, t, x, label=None, batch_size=None, n_steps=1,
                   time_discretization_type="linear"):
        from .time_discretization import time_grid

        grid = time_grid(n_steps, time_discretization_type)
        grid = s + (t - s) * grid if t >= s else t + (s - t) * grid[::-1]
        if not (torch.diff(torch.tensor(grid.copy(), dtype=x.dtype)).abs() > 0).all():
            raise ValueError("Time nodes collapse at the model's precision; use fewer steps")
        if batch_size is None:
            batch_size = x.shape[0]
        assert x.shape[0] % batch_size == 0
        n_batches = x.shape[0] // batch_size
        x = x.reshape((n_batches, batch_size, *x.shape[1:]))
        divergence_rescale = self.config.divergence_rescale

        x_list, dlogp_list = [], []
        div = None
        for batch_idx in range(n_batches):
            x_batch = x[batch_idx]
            dlogp_batch = torch.zeros((batch_size,), device=x.device)
            for left, right in zip(grid[:-1], grid[1:]):
                t_curr = x_batch.new_full((batch_size,), float(left))
                t_next = x_batch.new_full((batch_size,), float(right))
                dt = t_next - t_curr
                out = self.calc_phi(t_curr, t_next, x_batch, label=label, train=False, return_div=True)
                if isinstance(out, tuple):
                    phi, div = out
                else:
                    phi, div = out, None
                x_batch = x_batch + dt[:, None, None] * phi
                if div is not None:
                    div = div * divergence_rescale
                    div_batch = div.reshape((div.shape[0], -1)).sum(dim=-1)
                    dlogp_batch = dlogp_batch - dt * div_batch
            x_list.append(x_batch)
            dlogp_list.append(dlogp_batch)
        x = torch.cat(x_list, dim=0)
        dlogp = torch.cat(dlogp_list, dim=0)
        if div is None:
            return x, None
        return x, dlogp

    @torch.no_grad()
    def sample_forward(self, x0, label=None, batch_size=None, n_steps=1, return_nfe=True,
                       time_discretization_type="linear"):
        x1, dlogp1 = self._integrate(0.0, 1.0, x0, label=label, batch_size=batch_size,
                                    n_steps=n_steps, time_discretization_type=time_discretization_type)
        nfe = n_steps
        logp0 = self.base_log_prob(x0)
        if not self.dit.predict_divergence:
            if return_nfe:
                return self.unscale_samples(x1), None, nfe
            return self.unscale_samples(x1), None
        x1, dlogp1 = self.unscale_samples(x1, dlogp1)
        if return_nfe:
            return x1, logp0 + dlogp1, nfe
        return x1, logp0 + dlogp1

    @torch.no_grad()
    def sample_inverse(self, x1, label=None, batch_size=None, n_steps=1, return_nfe=True,
                       time_discretization_type="linear"):
        assert self.dit.predict_divergence, "predict_divergence must be True"
        x1, dlogp_rescale = self.rescale_samples(x1, torch.zeros((x1.shape[0],), device=x1.device))
        x0, dlogp0 = self._integrate(1.0, 0.0, x1, label=label, batch_size=batch_size,
                                    n_steps=n_steps, time_discretization_type=time_discretization_type)
        nfe = n_steps
        logp0 = self.base_log_prob(x0)
        logp1 = logp0 - dlogp0 - dlogp_rescale
        if return_nfe:
            return x0, logp0, logp1, nfe
        return x0, logp0, logp1
