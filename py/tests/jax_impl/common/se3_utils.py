import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

def _assert_shape(x: jnp.ndarray) -> None:
    assert len(x.shape) == 3 and x.shape[-1] == 3, f"Shape of x must be (N, n_atoms, 3), but got {x.shape}"

def remove_mean(x: jnp.ndarray) -> jnp.ndarray:
    """Remove mean from x."""
    _assert_shape(x)
    return x - x.mean(axis=1, keepdims=True)

def assert_mean_free(x: jnp.ndarray) -> None:
    _assert_shape(x)
    assert jnp.allclose(x.mean(axis=1, keepdims=True), 0.0, atol=1e-6), "x is not mean-free"

def interatomic_dist(samples, return_sq_norms=False, return_displacements=False,
                     keep_only_upper_tri=True, mask=None):
    """Pair-wise Euclidean distances between all atoms inside each configuration."""
    n_particles = samples.shape[-2]

    # Compute pair-wise displacement vectors (broadcasted subtraction)
    displacements = samples[:, None, :, :] - samples[:, :, None, :]

    # Keep only the strictly upper-triangular entries (i < j)
    if keep_only_upper_tri:
        if mask is None:
            mask = jnp.triu(
                jnp.ones((n_particles, n_particles), dtype=bool),
                k=1
            )
        displacements = displacements[:, mask]

    # Euclidean norm along the coordinate axis
    if return_sq_norms:
        sq_norms = jnp.sum(jnp.square(displacements), axis=-1)
        if return_displacements:
            return sq_norms, displacements
        else:
            return sq_norms
    else:
        distances = jnp.linalg.norm(displacements, axis=-1)
        if return_displacements:
            return distances, displacements
        else:
            return distances


def find_alignment_kabsch(P, Q) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Optimal rigid alignment of two point clouds via the Kabsch algorithm."""
    # Shift points w.r.t centroid
    centroid_P, centroid_Q = jnp.mean(P, axis=0), jnp.mean(Q, axis=0)
    P_c, Q_c = P - centroid_P, Q - centroid_Q
    
    # Find rotation matrix by Kabsch algorithm
    H = P_c.T @ Q_c
    U, S, Vt = jnp.linalg.svd(H, full_matrices=False)
    V = Vt.T
    
    # ensure right-handedness
    d = jnp.sign(jnp.linalg.det(V @ U.T))
    
    # Trick for jax.vmap
    dim = P.shape[-1]
    if dim == 3:
        diag_values = jnp.array([1.0, 1.0, d], dtype=P.dtype)
    elif dim == 2:
        diag_values = jnp.array([1.0, d], dtype=P.dtype)
    else:
        raise ValueError("Unsupported dim for kabsch")
        
    # This is only [[1,0,0],[0,1,0],[0,0,d]]
    M = jnp.eye(dim, dtype=P.dtype) * diag_values
    R = V @ M @ U.T
    
    # Find translation vectors
    t = centroid_Q[None, :] - (R @ centroid_P[None, :].T).T
    t = t.T
    
    return R, t.squeeze()


def calculate_rmsd(pos, ref):
    """Root-Mean-Square Deviation (RMSD) after optimal superposition."""
    if pos.shape[0] != ref.shape[0]:
        raise ValueError("pos and ref must have the same number of points")

    R, t = find_alignment_kabsch(ref, pos)  # rotate *ref* onto *pos*
    ref_aligned = (R @ ref.T).T + t  # apply rigid transform

    rmsd = jnp.mean(jnp.linalg.norm(ref_aligned - pos, axis=1))
    return rmsd


def calculate_rmsd_matrix(R_ref, R):
    """Pairwise RMSD matrix between two batches of structures."""
    # jax.vmap uses `in_axes` instead of `in_dims`
    fn_vmap_row = jax.vmap(calculate_rmsd, in_axes=(0, None))
    fn_vmap_row_col = jax.vmap(fn_vmap_row, in_axes=(None, 0))
    return fn_vmap_row_col(R, R_ref)


def avg_distance_to_origin(samples, n_particles, n_dimensions):
    """Average radial distance of particles from the origin after centring."""
    # Assuming remove_mean subtracts the center of mass
    # If you have a custom remove_mean, you can use it, otherwise here is the inline equivalent:
    samples = samples - jnp.mean(samples, axis=-2, keepdims=True)
    
    batch_size = samples.shape[0]  # assume leading dimension is the batch size
    samples = samples.reshape(batch_size, n_particles, n_dimensions)
    return jnp.mean(jnp.linalg.norm(samples, axis=-1), axis=1)


def compute_intersection(h1, h2):
    """Intersection between two histograms"""
    return jnp.sum(jnp.minimum(h1, h2))


def compute_correlation(h1, h2):
    """Correlation between two histograms"""
    h1_norm = h1 - jnp.mean(h1)
    h2_norm = h2 - jnp.mean(h2)
    return jnp.sum(h1_norm * h2_norm) / jnp.sqrt(
        jnp.sum(jnp.square(h1_norm)) * jnp.sum(jnp.square(h2_norm))
    )


def sample_random_rotation(key, n_samples):
    """Generates random 3D rotation matrices (SO(3)).
    
    Args:
        key: JAX random key
        n_samples: Number of samples to generate. Returns (n_samples, 3, 3).
    """
    keys = jax.random.split(key, n_samples)
    
    def _single_rot(k):
        # Uniform sampling on S^3 (unit quaternion sphere)
        # This guarantees a proper rotation (det = +1), never a reflection.
        u1, u2, u3 = jax.random.uniform(k, (3,))
        
        sqrt_1_minus_u1 = jnp.sqrt(1 - u1)
        sqrt_u1 = jnp.sqrt(u1)
        
        q0 = sqrt_1_minus_u1 * jnp.sin(2 * jnp.pi * u2)
        q1 = sqrt_1_minus_u1 * jnp.cos(2 * jnp.pi * u2)
        q2 = sqrt_u1 * jnp.sin(2 * jnp.pi * u3)
        q3 = sqrt_u1 * jnp.cos(2 * jnp.pi * u3)
        
        # Convert quaternion to rotation matrix
        # R = [ 1 - 2(q2^2 + q3^2)   2(q1q2 - q0q3)     2(q1q3 + q0q2)
        #       2(q1q2 + q0q3)     1 - 2(q1^2 + q3^2)   2(q2q3 - q0q1)
        #       2(q1q3 - q0q2)     2(q2q3 + q0q1)     1 - 2(q1^2 + q2^2) ]
        
        R = jnp.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        return R

    return jax.vmap(_single_rot)(keys)