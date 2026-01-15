import jax
import jax.numpy as jnp
from jax import jit
import optax


# ------------------------------------------------------------
# Hyperspherical parametrization (unit vectors by construction)
# ------------------------------------------------------------
def angles_to_unit_vectors(theta):
    """
    Convert hyperspherical angles to unit vectors (row-wise).

    Parameters
    ----------
    theta : array, shape (R, P-1)
        Each row contains the angular parameters of one unit vector in R^P.

    Returns
    -------
    A : array, shape (R, P)
        Each row is a unit-norm vector.
    """
    R, P_minus_1 = theta.shape
    P = P_minus_1 + 1

    components = []
    sin_prod = jnp.ones((R,))

    for p in range(P - 1):
        components.append(sin_prod * jnp.cos(theta[:, p]))
        sin_prod = sin_prod * jnp.sin(theta[:, p])

    components.append(sin_prod)  # last coordinate

    # Stack as columns, then transpose to rows
    return jnp.stack(components, axis=1)


# ------------------------------------------------------------
# Tensor reconstruction
# ------------------------------------------------------------
def reconstruct_tensor(A, C):
    """
    Z_hat[p,q,k] = sum_u A[u,p] A[u,q] C[k,u]
    """
    return jnp.einsum("up,uq,ku->pqk", A, A, C)


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
    tol = 1e-5
):
    """
    Fit Z_{p,q,k} ≈ sum_u A_{u,p} A_{u,q} C_{k,u}
    with ||A[u,:]|| = 1 enforced via angles.
    """
    P, _, K = Z.shape

    # --- initialize parameters ---
    if initial_guess is None:
        key, k1, k2 = jax.random.split(key, 3)
        theta = jax.random.uniform(k1, shape=(rank, P-1), minval=0.1, maxval=jnp.pi - 0.1)
        C = jax.random.normal(k2, shape=(K, rank))
        params = (theta, C)
    else:
        params = initial_guess
    
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


    # --- training loop ---
    for step in range(n_steps+1):
        params, opt_state, loss = train_step(params, opt_state, Z, rho)

        if step % 400 == 0:
            app_error = loss_fn(params, Z, 0)
            app_1norm = jnp.sum(jnp.abs(params[1]))
            print(f"Step {step:5d} | Approximation error = {app_error:.6e}, One-norm = {app_1norm:.6e}")

        if loss < tol:
            break

    theta, C = params

    return theta, C