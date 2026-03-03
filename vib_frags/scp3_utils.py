"""
scp3.py

A JAX and Optax powered module to perform Symmetric CP3 (SCP3) tensor factorization 
on 9-dimensional heavily symmetric target tensors using Algebraic Expansion.
"""

import functools
import time
import jax
import jax.numpy as jnp
import optax
import numpy as np

# Enforce double precision (critical for scientific computing)
jax.config.update("jax_enable_x64", True)

@functools.partial(jax.jit, static_argnames=['rank', 'nmodes', 'nmodals'])
def get_symmetric_factors(A_params_flat: jnp.ndarray, rank: int, nmodes: int, nmodals: int) -> jnp.ndarray:
    """
    Constructs the macro-factor matrix A while strictly enforcing the internal 
    modal symmetry A[i, p, q] = A[i, q, p]. Returns the flattened (rank, D) shape 
    for optimized tensor contraction in the loss function.
    """
    idx_i, idx_j = jnp.triu_indices(nmodals)
    
    A = jnp.zeros((rank, nmodes, nmodals, nmodals), dtype=jnp.float64)
    A = A.at[:, :, idx_i, idx_j].set(A_params_flat)
    A = A.at[:, :, idx_j, idx_i].set(A_params_flat)
    
    D = nmodes * nmodals**2
    return A.reshape((rank, D))



@functools.partial(jax.jit, static_argnames=['rank', 'nmodes', 'nmodals'])
def scp3_algebraic_loss(params: dict, trbt_3d: jnp.ndarray, trbt_norm_sq: float, 
                        rank: int, nmodes: int, nmodals: int) -> float:
    """
    Computes the squared Frobenius norm ||trbt - \\hat{trbt}||_F^2 algebraically.
    Expects trbt_3d to be the internally reshaped (D, D, D) tensor.
    """
    a = params['a']
    A_params = params['A']
    
    A = get_symmetric_factors(A_params, rank, nmodes, nmodals)
    
    # -2 * <trbt, \\hat{trbt}>
    trbtA1 = jnp.tensordot(A, trbt_3d, axes=([1], [0]))  
    trbtA2 = jnp.einsum('mab,ma->mb', trbtA1, A)             
    trbt_A = jnp.einsum('mb,mb->m', trbtA2, A)               
    term2 = -2.0 * jnp.dot(a, trbt_A)
    
    # ||\\hat{trbt}||_F^2
    G = jnp.dot(A, A.T)            
    G3 = G ** 3                    
    aa = jnp.outer(a, a)           
    term3 = jnp.sum(aa * G3)
    
    loss_sq = trbt_norm_sq + term2 + term3
    return jnp.maximum(loss_sq, 0.0)



def estimate_rank(trbt: jnp.ndarray, tol: float = 1e-8, multiplier: int = 2) -> int:
    """Estimates a good starting rank for scp3 based on the unfolding matrix rank."""
    # 1. Infer D from the 9D tensor and unfold directly to a (D, D^2) matrix
    D = trbt.shape[0] * trbt.shape[1]**2
    trbt_mat = trbt.reshape((D, D**2))
    
    # 2. Compute the D x D covariance matrix and its real eigenvalues
    eigenvalues = jnp.linalg.eigvalsh(jnp.dot(trbt_mat, trbt_mat.T))
    
    # 3. Count significant eigenvalues and apply the multiplier x heuristic
    R_mat = jnp.sum(eigenvalues > tol).item()
    
    print(f"Matrix rank (R_mat) at tolerance {tol}: {R_mat}")
    return int(multiplier * R_mat)



def fit_scp3(trbt: jnp.ndarray, rank: int | None = None, lr: float = 1e-2, 
             maxiter: int = 5000, initial_guess: dict = None, print_every: int = 500, seed: int = 42):
    """
    Optimizes the SCP3 factorization using JAX + Optax.
    
    Args:
        trbt: A 9D JAX array of shape 
              (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
        rank: Number of terms in the decomposition. If initial guess is provided, this can be None.
        lr: Optimizer learning rate.
        maxiter: Number of training steps.
        print_every: Frequency of progress output.
        seed: Random seed for initialization.
        
    Returns:
        a_final: The learned 1D weights array of shape (rank,)
        A_final: The learned 4D factors array of shape (nmodes, nmodals, nmodals, rank)
                 where A[i, p, q, r] == A[i, q, p, r]
        params: Raw parameter dictionary
    """
    # 1. Infer dimensions and validate 9D input shape
    if trbt.ndim != 9:
        raise ValueError(f"Expected a 9D tensor, got {trbt.ndim}D tensor.")
        
    nmodes = trbt.shape[0]
    nmodals = trbt.shape[1]
    
    expected_shape = (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
    if trbt.shape != expected_shape:
        raise ValueError(f"Expected trbt shape {expected_shape}, got {trbt.shape}")
    
    # 2. Reshape internally to (D, D, D) for math operations
    D = nmodes * nmodals**2
    trbt_3d = trbt.reshape((D, D, D))
    
    print("Precomputing target norm...")
    trbt_norm_sq = jnp.sum(trbt_3d ** 2)
    trbt_norm = jnp.sqrt(trbt_norm_sq)
    
    # 3. Initialize parameters
    K_params = nmodals * (nmodals + 1) // 2
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    if rank is None and initial_guess is None:
        raise ValueError("Either 'rank' must be specified or an 'initial_guess' must be provided.")
        
    if initial_guess is not None:
        a_init = initial_guess['a']
        A_params_init = initial_guess['A']
        rank_init = a_init.shape[0]
        if rank_init < rank: # Pad with random values if initial guess rank is smaller
            a_init = jnp.concatenate([a_init, jnp.zeros((rank - rank_init,), dtype=jnp.float64)])
            A_params_init = jnp.concatenate([A_params_init, jnp.zeros((rank - rank_init, nmodes, K_params), dtype=jnp.float64)], axis=0)
        elif rank_init > rank: # Truncate if initial guess rank is larger
            raise ValueError(f"Rank used by the initial guess ({rank_init}) is larger than the target rank ({rank}). Truncation is not supported. Please provide an initial guess with rank less than or equal to the target rank.")
    else:
        a_init = jax.random.normal(k1, (rank,), dtype=jnp.float64)
        A_params_init = jax.random.normal(k2, (rank, nmodes, K_params), dtype=jnp.float64)
    
    # 4. Set the correct rank
    if rank is None:
        # If rank was not provided, it must have been inferred from initial_guess
        rank = a_init.shape[0]

    A_init_matrix = get_symmetric_factors(A_params_init, rank, nmodes, nmodals)
    G_init = jnp.dot(A_init_matrix, A_init_matrix.T)
    term3_init = jnp.sum(jnp.outer(a_init, a_init) * (G_init ** 3))
    trbt_hat_norm_init = jnp.sqrt(jnp.maximum(term3_init, 1e-12))
    
    scale_factor = trbt_norm / trbt_hat_norm_init
    a_init = a_init * scale_factor
    
    params = {'a': a_init, 'A': A_params_init}
    
    # 5. Setup optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def step(params, opt_state):
        loss_val, grads = jax.value_and_grad(scp3_algebraic_loss)(
            params, trbt_3d, trbt_norm_sq, rank, nmodes, nmodals
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    # 6. Optimization Loop
    print(f"Starting SCP3 optimization for {maxiter} iterations (rank={rank})...", flush=True)
    start_time = time.time()
    
    for i in range(maxiter):
        params, opt_state, loss_sq_val = step(params, opt_state)
        
        if i % print_every == 0 or i == maxiter - 1:
            frob_error = jnp.sqrt(loss_sq_val)
            rel_error = frob_error / trbt_norm
            print(f"Iteration {i:5d} | Error Norm: {frob_error:.6e} | Relative Error: {rel_error:.6f}", flush=True)
            
    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
    
    # 7. Reshape final factors back to (rank, nmodes, nmodals, nmodals)
    A_final_flat = get_symmetric_factors(params['A'], rank, nmodes, nmodals)
    A_final_4d = A_final_flat.reshape((rank, nmodes, nmodals, nmodals))
    A_final_4d = np.transpose(np.array(A_final_4d), (1, 2, 3, 0))
    
    return np.array(params['a']), A_final_4d, params

# Example Usage Block
if __name__ == "__main__":
    nmodes = 50
    nmodals = 3
    rank = 100 
    
    expected_shape = (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
    print(f"Allocating dummy target tensor of shape {expected_shape}...")
    
    D = nmodes * nmodals**2
    key = jax.random.PRNGKey(99)
    dummy_A = jax.random.normal(key, (50, D), dtype=jnp.float64)
    dummy_trbt_3D = jnp.einsum('mi,mj,mk->ijk', dummy_A, dummy_A, dummy_A)
    dummy_trbt = dummy_trbt_3D.reshape(expected_shape)
    
    final_a, final_A, raw_params = fit_scp3(
        trbt=dummy_trbt, 
        rank=rank, 
        lr=0.01, 
        maxiter=2000, 
        print_every=200
    )
    
    print(f"Final Factor Shape: {final_A.shape}")