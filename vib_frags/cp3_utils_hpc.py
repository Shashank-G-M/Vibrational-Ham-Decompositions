import jax
import jax.numpy as jnp
from jax import jit
import optax
import numpy as np
from functools import partial


# ------------------------------------------------------------
# Convert paramteres from dictionary to 1D array
# ------------------------------------------------------------
def pack_to_flat_params(nmodes, thetas_dict, Cs_dict, Nthc_list):
    """
    Converts legacy dictionary-based parameters into flat 2D arrays,
    dynamically padding or truncating them to match the target Nthc_list.
    """
    theta_list = []
    C_list = []
    
    for i in range(nmodes):
        target_rank = Nthc_list[i]
        theta_i = thetas_dict[i]
        C_i = Cs_dict[i] 
        
        old_rank = theta_i.shape[1]
        P_minus_1 = theta_i.shape[0]
        K_dim = C_i.shape[0]
        
        # 1. Pad with zeros if the warm start rank is too small
        if old_rank < target_rank:
            pad_len = target_rank - old_rank
            
            theta_pad = jnp.zeros((P_minus_1, pad_len))
            theta_i = jnp.concatenate([theta_i, theta_pad], axis=1)
            
            C_pad = jnp.zeros((K_dim, pad_len))
            C_i = jnp.concatenate([C_i, C_pad], axis=1)
            
        # 2. Truncate if the warm start rank is somehow larger
        elif old_rank > target_rank:
            theta_i = theta_i[:, :target_rank]
            C_i = C_i[:, :target_rank]
            
        theta_list.append(theta_i)
        C_list.append(C_i)
        
    # Stack the lists into flat JAX arrays
    theta_flat = jnp.concatenate(theta_list, axis=1)
    C_flat = jnp.concatenate(C_list, axis=1)
    
    return theta_flat, C_flat


# ------------------------------------------------------------
# Hyperspherical parametrization (Batched) - Transposed
# ------------------------------------------------------------
@jax.vmap
def batched_angles_to_unit_vectors(theta_b):
    """
    Convert hyperspherical angles to unit vectors for a single mode.
    Vectorized over the 'mode' dimension.
    Input shape: (P - 1, R)
    Output shape: (P, R) - Columns are normalized unit vectors.
    """
    # Swap the unpacked shape variables
    P_minus_1, R = theta_b.shape
    P = P_minus_1 + 1

    components = []
    sin_prod = jnp.ones((R,))

    for p in range(P - 1):
        # Slice the row `p` across all `R` columns
        components.append(sin_prod * jnp.cos(theta_b[p, :]))
        sin_prod = sin_prod * jnp.sin(theta_b[p, :])

    components.append(sin_prod) 
    
    # Stack along axis=0 to produce a (P, R) matrix
    return jnp.stack(components, axis=0)



# ------------------------------------------------------------
# Tensor reconstruction (Batched)
# ------------------------------------------------------------
@jax.vmap
def batched_reconstruct_tensor(A_b, C_b):
    """
    Reconstruct the target tensor for a single mode.
    Vectorized over the 'mode' dimension.
    C_b is assumed to be shape (K, rank).
    """
    return jnp.einsum("pu,qu,ku->pqk", A_b, A_b, C_b)


# ------------------------------------------------------------
# Loss Function (Corrected for Static Dimensions)
# ------------------------------------------------------------
@partial(jit, static_argnums=(4, 5, 6, 7))
def loss_fn(flat_params, Z_batched, mode_idx, rank_idx, nmodes, max_rank, P, K, rho):
    """
    Evaluates the loss across all modes simultaneously using scattered dense tensors.
    P and K are passed statically to avoid tracer leaks during JIT compilation.
    """
    theta_flat, C_flat = flat_params
    
    # 1. Initialize empty batched parameter tensors
    theta_big = jnp.zeros((nmodes, P-1, max_rank))
    C_big = jnp.zeros((nmodes, K, max_rank))
    
    # 2. Differentiable scatter operation
    theta_big = theta_big.at[mode_idx, :, rank_idx].set(theta_flat.T)
    C_big = C_big.at[mode_idx, :, rank_idx].set(C_flat.T)
    
    # 3. Batched forward pass
    A_big = batched_angles_to_unit_vectors(theta_big)
    Z_hat_batched = batched_reconstruct_tensor(A_big, C_big)
    
    # 4. Global loss computation
    recon_loss = 0.5 * jnp.sum((Z_hat_batched - Z_batched) ** 2)
    l1_penalty = rho * jnp.sum(jnp.abs(C_flat))
    
    return recon_loss + l1_penalty

# ------------------------------------------------------------
# Helper: Build Static Index Mapping
# ------------------------------------------------------------
def build_index_arrays(Nthc_list):
    """
    Generates 1D arrays mapping the flat parameter space to 
    the (mode, rank) coordinates of the dense tensor.
    """
    mode_idx = []
    rank_idx = []
    for mode, rank in enumerate(Nthc_list):
        mode_idx.extend([mode] * rank)
        rank_idx.extend(list(range(rank)))
        
    return jnp.array(mode_idx), jnp.array(rank_idx)

# ------------------------------------------------------------
# Driver Routine for Batched Fitting (Corrected State Tracking)
# ------------------------------------------------------------
def fit_cp_batched(
    Z_batched,
    Nthc_list,
    n_steps,
    lr,
    rho,
    initial_flat_params,
    tol=1e-5,
    eval_every=1000,
):
    nmodes, P, _, K = Z_batched.shape
    max_rank = max(Nthc_list)
    mode_idx, rank_idx = build_index_arrays(Nthc_list)
    
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(initial_flat_params)
    
    @jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(
            params, Z_batched, mode_idx, rank_idx, nmodes, max_rank, P, K, rho
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    params = initial_flat_params
    best_loss = float('inf')
    best_params = params
    
    for step in range(n_steps + 1):
        params, opt_state, loss = train_step(params, opt_state)
        current_loss = float(loss)
        
        # Track the best parameters before a potential NaN or divergence
        if current_loss < best_loss and not jnp.isnan(current_loss):
            best_loss = current_loss
            best_params = params
            
        if step % eval_every == 0:
            app_1norm = float(jnp.sum(jnp.abs(params[1])))
            print(f"Step {step:6d} | Global Loss = {current_loss:.6e} | C-Norm = {app_1norm:.6e}", flush=True)
            
            if current_loss < tol:
                print(f"Convergence tolerance reached at step {step}.")
                break
                
            if jnp.isnan(current_loss):
                print(f"Warning: NaN encountered at step {step}. Reverting to best parameters.")
                break

    return best_params