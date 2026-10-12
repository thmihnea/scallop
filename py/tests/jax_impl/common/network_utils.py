"""
Nicholas M. Boffi
10/5/25

Helper routines for neural network definitions.
"""

from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict
from tqdm import trange

from . import dit as dit
from jax.experimental.ode import odeint


class DiTFlowMap(nn.Module):
    config: config_dict.ConfigDict

    def setup(self):
        self.dit = dit.FlowMapAtomDiT(
            n_particles=self.config.n_particles,
            n_dimension=self.config.n_dimension,
            atom_type_labels=self.config.atom_type_labels,
            bonds=self.config.bonds,
            predict_divergence=self.config.predict_divergence,
            vectorized_divergence=self.config.vectorized_divergence,
            **self.config.dit_kwargs,
        )
        self.sigma_data = self.config.sigma_data
        self.use_weight = self.config.use_weight
        self.logvar_channels = self.config.logvar_channels
        self.use_bfloat16 = self.config.use_bfloat16

        # Single weight function - using positional embeddings
        self.logvar_fourier_s = dit.PositionalEmbedding(self.logvar_channels)
        self.logvar_fourier_t = dit.PositionalEmbedding(self.logvar_channels)

        self.logvar_linear = nn.Dense(1)

    def calc_weight(self, s: float, t: float) -> jnp.ndarray:
        """Calculate weight function."""
        if not self.use_weight:
            return 1.0

        s = jnp.asarray(s, dtype=jnp.float32)
        t = jnp.asarray(t, dtype=jnp.float32)

        embed_s = self.logvar_fourier_s(s).reshape(-1, self.logvar_channels)
        embed_t = self.logvar_fourier_t(t).reshape(-1, self.logvar_channels)
        
        # mp_sum equivalent for JAX without custom MP ops
        embed = (embed_s + embed_t) / jnp.sqrt(2.0)
        
        logvar = self.logvar_linear(embed).reshape(-1, 1, 1)
        return logvar

    def process_input(
        self, s: float, t: float, x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, bool]:
        """Process inputs for the model using s_dt parameterization."""
        # Ensure s and t are at least 1D so they can broadcast correctly
        s = jnp.asarray(s, dtype=jnp.float32)
        t = jnp.asarray(t, dtype=jnp.float32)
        if x.ndim == 2:
            x = x[None, ...]
            is_single = True
        elif x.ndim == 3:
            is_single = False
        else:
            raise ValueError(f"Unsupported x shape: {x.shape}")

        batch_size = x.shape[0]
        s = jnp.broadcast_to(s.reshape(-1, 1, 1), (batch_size, 1, 1))
        t = jnp.broadcast_to(t.reshape(-1, 1, 1), (batch_size, 1, 1))
        
        return s, t, x, is_single

    def calc_phi(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        init_weights: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
        # process_input returns ss, ts, xs
        ss, ts, xs, is_single = self.process_input(s, t, x)

        c_out = self.sigma_data
        c_in = 1.0 / self.sigma_data

        # Run the model
        xs_in = (c_in * xs).astype(dtype)
        # FlowMapAtomEGNN signature: __call__(self, x, ss, dt, train=False)
        phi_raw, div_raw = self.dit(xs_in, ss, ts - ss, train=train)
        
        phi_st = c_out * phi_raw.astype(jnp.float32)
        div_st = c_out * div_raw.astype(jnp.float32) if div_raw is not None else None

        if init_weights:
            # During initialization, ensure weight params are created
            _ = self.calc_weight(s, t)  # calc_weight expects the raw s, t values

        if calc_weight:
            logvar = self.calc_weight(s, t)[0]
            if return_div and div_st is not None:
                if is_single:
                    return phi_st[0], div_st[0], logvar[0]
                return phi_st, div_st, logvar
            if is_single:
                return phi_st[0], logvar[0]
            return phi_st, logvar

        if return_div and div_st is not None:
            if is_single:
                return phi_st[0], div_st[0]
            return phi_st, div_st
        if is_single:
            return phi_st[0]
        return phi_st

    def calc_b(
        self,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        return self.calc_phi(
            t,
            t,
            x,
            label=label,
            train=train,
            calc_weight=calc_weight,
            return_div=return_div,
        )

    def __call__(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        phi_rslt = self.calc_phi(
            s,
            t,
            x,
            label=label,
            train=train,
            calc_weight=calc_weight,
            init_weights=init_weights,
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
        Xst = x + (t - s) * phi_st

        # estimate uncertainty if requested
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

    def base_log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Log prob of mean-free N(0, diag(rescale^2)) for base distribution."""
        assert x.ndim == 3 and x.shape[2] == 3, "x must be of shape (batch, n_atoms, 3)"
        sigma = jnp.asarray(self.config.rescale)
        if sigma.ndim == 0:
            sigma = jnp.ones(x.shape[1:]) * sigma
        dims = tuple(range(1, x.ndim))
        quad = jnp.sum((x / sigma) ** 2, axis=dims)
        total_dim = float(x.shape[1] * x.shape[2])
        log_det = jnp.sum(jnp.log(sigma**2))
        log_norm = 0.5 * (total_dim * jnp.log(2 * jnp.pi) + log_det)
        unconstrained_log_prob = -0.5 * quad - log_norm    
        var_center_of_mass = jnp.mean(sigma**2, axis=0)  # Shape: (3,)
        correction = 0.5 * (3 * jnp.log(2 * jnp.pi) + jnp.sum(jnp.log(var_center_of_mass)))
        return unconstrained_log_prob + correction

    def rescale_samples(self, x: jnp.ndarray, logp: jnp.ndarray | None = None) -> jnp.ndarray:
        x = x / self.config.unscale_factor
        if logp is not None:
            logp = logp + self.config.n_particles * self.config.n_dimension * jnp.log(self.config.unscale_factor)
            return x, logp
        return x

    def unscale_samples(self, x: jnp.ndarray, logp: jnp.ndarray | None = None) -> jnp.ndarray:
        x = x * self.config.unscale_factor
        if logp is not None:
            logp = logp - self.config.n_particles * self.config.n_dimension * jnp.log(self.config.unscale_factor)
            return x, logp
        return x

    def _integrate(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        batch_size: int = None,
        n_steps: int = 1,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Integrate the model from s to t with n_steps.
        If s < t, integrate forward in time, starting from xs=x and logp_xs=logp_x.
        If s > t, integrate backward in time, starting from xt=x and logp_xt=logp_x.
        """
        dt = (t - s) / n_steps
        if batch_size is None:
            batch_size = x.shape[0]
        assert x.shape[0] % batch_size == 0, "x.shape[0] must be divisible by batch_size"
        n_batches = x.shape[0] // batch_size
        x = x.reshape((n_batches, batch_size, *x.shape[1:]))
        if label is not None:
            label = label.reshape((n_batches, batch_size, *label.shape[1:]))

        divergence_rescale = getattr(self.config, "divergence_rescale")
        x_list = []
        dlogp_list = []
        for batch_idx in trange(n_batches):
            t_curr = s * jnp.ones((batch_size,), dtype=jnp.float32)
            x_batch = x[batch_idx]
            dlogp_batch = jnp.zeros((batch_size,))
            if label is not None:
                label_batch = label[batch_idx]
            else:
                label_batch = None
            for _ in range(n_steps):
                t_next = t_curr + dt
                # Keep argument order identical to logging.py: calc_phi(s=t_curr, t=t_next, ...)
                out = self.calc_phi(
                    s=t_curr,
                    t=t_next,
                    x=x_batch,
                    label=label_batch,
                    train=False,
                    return_div=True,
                )
                if isinstance(out, tuple):
                    phi, div = out
                else:
                    phi = out
                    div = None
                x_batch = x_batch + dt * phi
                if div is not None:
                    div = div * divergence_rescale
                    div_batch = div.reshape((div.shape[0], -1)).sum(axis=-1)
                    dlogp_batch = dlogp_batch - dt * div_batch
                t_curr = t_next
            x_list.append(x_batch)
            dlogp_list.append(dlogp_batch)
        x = jnp.concatenate(x_list, axis=0)
        dlogp = jnp.concatenate(dlogp_list, axis=0)

        if div is None:
            return np.asarray(x), None

        return np.asarray(x), np.asarray(dlogp)

    def _integrate_dopri5(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        batch_size: int = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        return_nfe: bool = False,
    ) -> Tuple:
        """
        Integrate the model from s to t using the Dopri5 adaptive step size ODE solver.
        Batching matches ``_integrate``: if ``batch_size`` is None, the full leading
        dimension is one solve; otherwise ``x.shape[0]`` must be divisible by
        ``batch_size`` and Diffrax runs one solve per chunk.
        """
        import diffrax

        divergence_rescale = getattr(self.config, "divergence_rescale")

        def vector_field(t_curr, y, args):
            x_curr, dlogp_curr = y
            label_batch = args

            dt_eps = 1e-4 if t > s else -1e-4

            phi, div = self.calc_phi(
                s=t_curr,
                t=t_curr + dt_eps,
                x=x_curr,
                label=label_batch,
                train=False,
                return_div=True,
            )

            dx_dt = phi
            if div is not None:
                div = div * divergence_rescale
                ddlogp_dt = -div.reshape((div.shape[0], -1)).sum(axis=-1)
            else:
                ddlogp_dt = jnp.zeros_like(dlogp_curr)

            return (dx_dt, ddlogp_dt)

        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

        n = x.shape[0]
        if batch_size is None:
            batch_size = n
        assert n % batch_size == 0, "x.shape[0] must be divisible by batch_size"

        x_parts = []
        dlogp_parts = []
        num_steps_max = 0
        for start in range(0, n, batch_size):
            x_b = x[start : start + batch_size]
            if label is not None:
                label_b = label[start : start + batch_size]
            else:
                label_b = None
            dlogp_init = jnp.zeros((x_b.shape[0],))
            y0 = (x_b, dlogp_init)
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=s,
                t1=t,
                dt0=None,
                y0=y0,
                args=label_b,
                stepsize_controller=stepsize_controller,
            )
            x_parts.append(sol.ys[0][-1])
            dlogp_parts.append(sol.ys[1][-1])
            ns = int(np.asarray(sol.stats["num_steps"]))
            num_steps_max = max(num_steps_max, ns)

        x_final = jnp.concatenate(x_parts, axis=0)
        dlogp_final = jnp.concatenate(dlogp_parts, axis=0)

        if dlogp_final is None or jnp.all(dlogp_final == 0):
            dlogp_final = None
        else:
            dlogp_final = np.asarray(dlogp_final)

        if return_nfe:
            return np.asarray(x_final), dlogp_final, np.asarray(num_steps_max)

        return np.asarray(x_final), dlogp_final

    def sample_forward(
        self,
        x0: jnp.ndarray,
        label: float = None,
        batch_size: int = None,
        n_steps: int = 1,
        return_nfe: bool = True,
    ) -> Tuple:
        """
        Sample forward from x0 to x1.
        If n_steps is None, use Dopri5 to integrate.
        """
        if n_steps is None:
            res = self._integrate_dopri5(
                s=0.0,
                t=1.0,
                x=x0,
                label=label,
                batch_size=batch_size,
                return_nfe=True,
            )
            x1, dlogp1, nfe = res
        else:
            x1, dlogp1 = self._integrate(
                s=0.0,
                t=1.0,
                x=x0,
                label=label,
                batch_size=batch_size,
                n_steps=n_steps,
            )
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

    def sample_inverse(
        self,
        x1: jnp.ndarray,
        label: float = None,
        batch_size: int = None,
        n_steps: int = 1,
        return_nfe: bool = True,
    ) -> Tuple:
        """
        Sample inverse from x1 to x0.
        If n_steps is None, use Dopri5 to integrate.
        """
        assert self.dit.predict_divergence, "predict_divergence must be True"
        x1, dlogp_rescale = self.rescale_samples(x1, jnp.zeros((x1.shape[0],)))
        if n_steps is None:
            res = self._integrate_dopri5(
                s=1.0,
                t=0.0,
                x=x1,
                label=label,
                batch_size=batch_size,
                return_nfe=True,
            )
            x0, dlogp0, nfe = res
        else:
            x0, dlogp0 = self._integrate(
                s=1.0,
                t=0.0,
                x=x1,
                label=label,
                batch_size=batch_size,
                n_steps=n_steps,
            )
            nfe = n_steps
            
        logp0 = self.base_log_prob(x0)
        logp1 = logp0 - dlogp0 - dlogp_rescale
        if return_nfe:
            return x0, logp0, logp1, nfe
        return x0, logp0, logp1

def get_act(
    config: config_dict.ConfigDict,
) -> Callable:
    """Get the activation function for the network.

    Args:
        config: Configuration dictionary.
    """
    if config.act == "gelu":
        return jax.nn.gelu
    elif config.act == "swish" or config.act == "silu":
        return jax.nn.silu
    else:
        raise ValueError(f"Activation function {config.activation} not recognized.")


def setup_network(
    network_config: config_dict.ConfigDict,
) -> nn.Module:
    """Setup the neural network for the system.

    Args:
        config: Configuration dictionary.
    """
    return DiTFlowMap(config=network_config)