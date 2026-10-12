import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange


def bar(fwd_work, bwd_work, max_iter=10000, tol=1e-6):
    N_f = fwd_work.shape[0]
    N_b = bwd_work.shape[0]
    M = jnp.log(N_f / N_b)

    fwd_free_energy = -jax.nn.logsumexp(-fwd_work) + jnp.log(N_f)
    bwd_free_energy = jax.nn.logsumexp(-bwd_work) - jnp.log(N_b)
    dF = 0.5 * (fwd_free_energy + bwd_free_energy)
    
    for i in trange(max_iter):
        log_sum_bwd = jax.nn.logsumexp(jax.nn.log_sigmoid(-bwd_work - dF + M))
        log_sum_fwd = jax.nn.logsumexp(jax.nn.log_sigmoid(-fwd_work + dF - M))
        dF_ = dF + log_sum_bwd - log_sum_fwd
        
        if jnp.abs(dF_ - dF) < tol:
            break
        dF = dF_
        
    return dF, fwd_free_energy, bwd_free_energy