import jax
import jax.numpy as jnp
from jax import jit
import optax
import numpy as np
from collections import deque
from copy import copy


# ------------------------------------------------------------
# Hyperspherical parametrization (unit vectors by construction)
# ------------------------------------------------------------
def angles_to_unit_vectors(theta):
    """
    Convert hyperspherical angles to unit vectors.

    Parameters
    ----------
    theta : array, shape (P-1, R)
        Angular parameters for R unit vectors in R^P.

    Returns
    -------
    A : array, shape (P, R)
        Columns are unit-norm vectors.
    """
    # Compute all sines and cosines simultaneously
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)

    # Prefix products of sines along the coordinate axis (axis 0)
    # This replaces the Python loop computing sin_prod
    cumprod_sin = jnp.cumprod(sin_theta, axis=0)

    # The factors multiplying the cosines are 1 for the first coordinate,
    # and the cumulative product of sines for the subsequent ones.
    # We prepend a row of 1s and drop the last row of the cumprod to shift it.
    ones = jnp.ones((1, theta.shape[1]), dtype=theta.dtype)
    sin_factors = jnp.concatenate([ones, cumprod_sin[:-1, :]], axis=0)

    # Calculate the first P-1 coordinates
    comps_first = sin_factors * cos_theta

    # The last coordinate is the product of all sines (the very last row of cumprod)
    last_comp = cumprod_sin[-1:, :]

    # Concatenate along axis 0 to get final shape (P, R)
    return jnp.concatenate([comps_first, last_comp], axis=0)


# ------------------------------------------------------------
# Tensor reconstruction
# ------------------------------------------------------------
def reconstruct_tensor(A, C):
    """
    Z_hat[p,q,k] = sum_u A[p,u] A[q,u] C[k,u]
    """
    return jnp.einsum("pu,qu,ku->pqk", A, A, C)


# ------------------------------------------------------------
# Loss function
# ------------------------------------------------------------
def loss_fn(params, Z, rho):
    """
    Squared Frobenius norm loss.
    """
    theta, C = params
    A = angles_to_unit_vectors(theta)
    Z_hat = reconstruct_tensor(A, C)
    return 0.5 * jnp.sum((Z_hat - Z) ** 2) + rho*jnp.sum(jnp.abs(C))



# ------------------------------------------------------------
# Fitting routine
# ------------------------------------------------------------
def fit_cp_symmetric(
    Z,
    rank,
    n_steps=10000,
    lr=5e-3,
    rho=0,
    key=jax.random.PRNGKey(0),
    initial_guess = None,
    tol = 1e-5,
    check_convergence = False,
    window_size = 100
):
    """
    Fit Z[p,q,k] = sum_u A[p,u] A[q,u] C[k,u]
    with ||A[:,u]|| = 1 enforced via angles.
    """
    P, _, K = Z.shape

    # --- initialize parameters ---
    if initial_guess is None:
        key, k1, k2 = jax.random.split(key, 3)
        theta = jax.random.uniform(k1, shape=(P-1,rank), minval=0.1, maxval=jnp.pi - 0.1)
        C = jax.random.normal(k2, shape=(K, rank))*0
    else:
        params = initial_guess
        theta_ = params[0]
        C_ = params[1]
        if theta_.shape[1] < rank:
            theta = jnp.zeros((P-1,rank))
            theta = theta.at[:, :theta_.shape[1]].set(theta_)
            C = jnp.zeros((K, rank))
            C = C.at[:, :C_.shape[1]].set(C_)
        else:
            theta = theta_[:, :rank]
            C = C_[:, :rank]
    params = (jnp.array(theta), jnp.array(C))
    
    # --- Adam optimizer ---
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    
    # ------------------------------------------------------------
    # Single Adam update step
    # ------------------------------------------------------------
    @jit
    def train_step(params, opt_state, Z, rho):
        """
        One Adam step.
        """
        loss, grads = jax.value_and_grad(loss_fn)(params, Z, rho)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # 1. Initialize the buffer with a fixed size given by window_size
    loss_history = deque(maxlen=window_size)
    # --- training loop ---
    for step in range(100):
        params, opt_state, loss = train_step(params, opt_state, Z, rho)

    # --- training loop ---
    for step in range(n_steps+1):
        params, opt_state, loss = train_step(params, opt_state, Z, rho)

        if step % 400 == 0:
            app_error = loss_fn(params, Z, 0)
            app_1norm = jnp.sum(jnp.abs(params[1]))
            print(f"Step {step:5d} | Approximation error = {app_error:.6e}, One-norm = {app_1norm:.6e}")

        loss_history.append(float(loss))
        if step > 2*window_size:
            if check_convergence:
                if np.std(np.abs(loss_history)) < 1e-7:
                    print(f"Loss stabilized at step {step}")
                    break
            else:
                if app_error < tol:
                    break

    theta, C = params

    return np.array(theta), np.array(C)







# ------------------------------------------------------------
# Greedy loss function (experimental)
# ------------------------------------------------------------
def greedy_loss_fn(theta_u, Z_residual, rho):
    """
    Squared Frobenius norm loss.
    """
    A_u = angles_to_unit_vectors(theta_u)
    C_u = jnp.einsum("p,q,pqk->k", A_u[0], A_u[0], Z_residual)
    return -jnp.sum(jnp.abs(C_u))                                       #Negative sign is there to ensure that optimizer maximizes one-norm of C_u



#Experimental!
def fit_greedy_cp_symmetric(
    Z,
    rank,
    n_steps=10000,
    lr=5e-3,
    rho=0,
    key=jax.random.PRNGKey(0),
    initial_guess = None,
    tol = 1e-5,
    check_convergence = False,
    window_size = 100
):
    """
    Fit Z_{p,q,k} ≈ sum_u A_{p,u} A_{q,u} C_{k,u}
    with ||A[u,:]|| = 1 enforced via angles.
    """
    P, _, K = Z.shape

    # --- initialize parameters ---
    if initial_guess is None:
        key, k1, k2 = jax.random.split(key, 3)
        theta = jax.random.uniform(k1, shape=(P-1,rank), minval=0.1, maxval=jnp.pi - 0.1)
    else:
        params = initial_guess
        theta_ = params[0]
        if theta_.shape[1] < rank:
            theta = jnp.zeros((P-1,rank))
            theta = theta.at[:, :theta_.shape[1]].set(theta_)
        else:
            theta = theta_
    
    # --- Adam optimizer ---
    optimizer = optax.adam(lr)
    
    # ------------------------------------------------------------
    # Single Adam update step
    # ------------------------------------------------------------
    @jit
    def train_step(params, opt_state, Z_residual, rho):
        """
        One Adam step.
        """
        loss, grads = jax.value_and_grad(greedy_loss_fn)(params, Z_residual, rho)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    final_theta = np.zeros((P-1,rank))
    final_C = np.zeros((Z.shape[-1], rank))

    Z_residual = copy(Z)
    for u in range(rank):
        theta_u = theta[u]
        opt_state = optimizer.init(theta_u)
        
        # --- training loop ---
        for step in range(n_steps+1):
            theta_u, opt_state, loss = train_step(theta_u, opt_state, Z_residual, rho)

            if step % 400 == 0:
                if np.abs(loss) < tol:
                    break
        
        final_theta[u] = theta_u
        A_u = angles_to_unit_vectors(theta_u.reshape(1, -1))
        C_u = jnp.einsum("p,q,pqk->k", A_u[0], A_u[0], Z_residual)
        final_C[:, u] = np.array(C_u)

        Z_approx_u = jnp.einsum("p,q,k->pqk", A_u[0], A_u[0], C_u)
        Z_residual = Z_residual - Z_approx_u
        error = 0.5*float(jnp.sqrt(jnp.sum((Z_residual)**2)))
        print (f"Rank: {u}, Error: {error}")

    return np.array(final_theta), np.array(final_C)