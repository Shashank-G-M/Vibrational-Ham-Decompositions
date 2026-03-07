import numpy as np
import jax
from jax import numpy as jnp
from jax import grad, jit, vmap
from jax import lax
import optax
from functools import partial
import numpy as np
from copy import copy
from .tensor_utils import symmetrize_tbt, symmetrize_trbt

# Forcibly disable 64-bit execution
jax.config.update("jax_enable_x64", True)

# Optional: Force matmul operations to use the fastest possible float32 math
# jax.config.update("jax_default_matmul_precision", "tensorfloat32")


def _get_BLISS_sizes(num_ob_syms, Norbs):
    avec_len = num_ob_syms
    bvec_len = int(num_ob_syms * (num_ob_syms+1)/2)
    ob_mat_num_params = int(Norbs*(Norbs+1)/2)
    dvec_len = int(num_ob_syms * (num_ob_syms-1)/2)

    return avec_len, bvec_len, ob_mat_num_params, dvec_len



def _symmetrize_zeta(zeta):
    zeta_sym = (
        zeta + 
        np.transpose(zeta, (1,0,3,2))) / 2.0
    return zeta_sym



def _symmetrize_gamma(gamma):
    gamma_sym = (
        gamma + 
        np.transpose(gamma, (0,2,1,3,5,4)) +
        np.transpose(gamma, (1,0,2,4,3,5)) +
        np.transpose(gamma, (1,2,0,4,5,3)) +
        np.transpose(gamma, (2,0,1,5,3,4)) +
        np.transpose(gamma, (2,1,0,5,4,3))
    ) / 6.0
    return gamma_sym




def unfold_vib_hthc(theta, zeta, gamma=None, trbt_is_None=True):
    Nmode = len(theta)
    Nmodal = theta[0].shape[-1] + 1

    tbt_thc = np.zeros((Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal))
    xupq = {}
    for i, theta_i in theta.items():
        xi_i = angles_to_unit_vectors(theta_i.T)
        xi_i = xi_i.T
        xupq[i] = np.einsum("Up,Uq->Upq", xi_i, xi_i)

    for (i,j), Zij in zeta.items():
        tbt_thc[i,:,:,j,:,:] = np.einsum('UV,Upq,Vrs->pqrs', Zij, xupq[i], xupq[j])

    if not trbt_is_None:
        trbt_thc = np.zeros((Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal))
    else:
        trbt_thc = None
    return tbt_thc, trbt_thc






@partial(jax.jit, static_argnums=())
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

# Assuming angles_to_unit_vectors takes a 2D array and returns a 2D array
# We vmap it so it can process the entire 3D theta array at once.
# in_axes=(0,) means map over the M dimension.
batched_angles_to_vectors = vmap(angles_to_unit_vectors, in_axes=(0,))




def tbt_error(theta, zeta, tbt_full):
    xupq = {}
    for i, theta_i in theta.items():
        xi_i = angles_to_unit_vectors(theta_i)
        xupq[i] = np.einsum("pU,qU->Upq", xi_i, xi_i)
    loss = 0
    for (i,j), Zij in zeta.items():
        loss += np.sum((tbt_full[i,:,:,j,:,:] - np.einsum('UV,Upq,Vrs->pqrs', Zij, xupq[i], xupq[j]))**2)
    tot_cost = np.sqrt(loss)
    return tot_cost




@partial(jax.jit, static_argnums=()) 
def _tbt_error(theta, zeta, tbt_full):
    # theta shape: (M, N-1, R)
    # zeta shape:  (M, M, R, R) -> axes: (i, j, U, V)
    
    # 1. Vectorized generation of xupq  
    # Apply function across all M blocks simultaneously
    xi = batched_angles_to_vectors(theta) 
    
    # Create xupq for all M blocks at once. 
    xupq = jnp.einsum("MpU,MqU->MUpq", xi, xi, optimize=True)

    # 2. First Contraction: Contract U between zeta and xupq
    # zeta: (i, j, U, V), xupq: (i, U, p, q) -> intermediate: (i, j, V, p, q)
    intermediate = jnp.einsum('ijUV,iUpq->ijVpq', zeta, xupq, optimize=True)

    # 3. Second Contraction: Contract V between intermediate and xupq
    # intermediate: (i, j, V, p, q), xupq: (j, V, r, s) 
    # We output to (i, p, q, j, r, s) to perfectly match the layout of tbt_full!
    recon = jnp.einsum('ijVpq,jVrs->ipqjrs', intermediate, xupq, optimize=True)

    

    # 4. Global Loss Calculation
    # We subtract the entire reconstructed 6D tensor from the target at once.
    loss = jnp.sqrt(jnp.sum((tbt_full - recon)**2))
    
    return loss


@partial(jax.jit, static_argnums=()) 
def _tbt_error_opt(theta, zeta, tbt_full, norm_tbt_sq):
    # theta shape: (M, N-1, R)
    # zeta shape:  (M, M, R, R) -> axes: (i, j, U, V)
    
    # 1. Vectorized generation of xupq
    xi = batched_angles_to_vectors(theta) 
    xupq = jnp.einsum("MpU,MqU->MUpq", xi, xi, optimize=True)

    # 2. Flatten spatial dimensions for efficient BLAS execution
    M, R, p, q = xupq.shape
    P = p * q
    x_flat = xupq.reshape(M, R, P)
    
    # Flatten target tensor: (M, p, q, M, r, s) -> (M, P, M, P)
    tbt_flat = tbt_full.reshape(M, P, M, P)

    # 3. Term 2: The Cross Term (-2 <T, R>)
    # Project T onto the x basis step-by-step to avoid large memory allocations
    Y = jnp.einsum('iPjQ,iUP->ijUQ', tbt_flat, x_flat, optimize=True)
    W = jnp.einsum('ijUQ,jVQ->ijUV', Y, x_flat, optimize=True)
    cross_term = jnp.einsum('ijUV,ijUV->', zeta, W, optimize=True)

    # 4. Term 3: The Norm of the Reconstruction (||R||^2)
    # Calculate the spatial overlap (Gram) matrix first
    S = jnp.einsum('iUP,iAP->iUA', x_flat, x_flat, optimize=True)
    
    # Contract overlaps with zeta. 
    # Using Z_inter controls the contraction path strictly to smaller intermediate ranks
    Z_inter = jnp.einsum('ijUV,jVB->ijUB', zeta, S, optimize=True)
    R_norm_sq = jnp.einsum('ijUB,ijAB,iUA->', Z_inter, zeta, S, optimize=True)

    # 5. Global Loss Calculation
    # ||T - R||^2 = ||T||^2 - 2<T,R> + ||R||^2
    loss_sq = norm_tbt_sq - 2.0 * cross_term + R_norm_sq
    
    # jnp.maximum(..., 0.0) is mathematically required here. 
    # Due to floating point math, a near-perfect reconstruction might yield -1e-12,
    # and taking the square root of a negative float causes a NaN kernel crash.
    loss = jnp.sqrt(jnp.maximum(loss_sq, 0.0))
    
    return loss



def ten_norm(ten, fro=True):
    """
    Input tensor can be numpy or jaxnumpy. If fro=True, L2 norm of the flatenned array (Frobenius norm) is returned, 
    otherwise L1 norm of the flattened array is returned.
    """
    if fro == True:
        return jnp.sqrt(jnp.sum(ten**2))
    else:
        return jnp.sum(jnp.abs(ten))







def hthc_vib_one_norm(kappa, tbt_full, trbt_full, zeta, gamma, obt_is_none, trbt_is_None, Rthc):
    #This function assumes the input zeta is a dictionary with keys as tuples of mode indices
    lambda_1 = 0.0
    sym_tbt_full = symmetrize_tbt(tbt_full)
    if trbt_is_None == False:
        sym_trbt_full = symmetrize_trbt(trbt_full)

    if obt_is_none == False:
        kappa = kappa + (1)*np.einsum('ipqjrr -> ipq', sym_tbt_full)
        if trbt_is_None == False:
            kappa += (3/4)*np.einsum('ipqjrrktt -> ipq', sym_trbt_full)
        Nmode = kappa.shape[0]
        for i in range (Nmode):                                 #Diagonalize obt in each mode
            kappa_i = kappa[i]
            D = np.linalg.eigvalsh(kappa_i)
            lambda_1 += (1/2)*np.sum(np.abs(D))

    zeta_tilde = np.zeros((Nmode, Nmode, Rthc, Rthc))
    for (i,j), Zij in zeta.items():
        u_max, v_max = Zij.shape
        zeta_tilde[i,j,:u_max, :v_max] = Zij
    zeta_tilde = _symmetrize_zeta(zeta_tilde)

    lambda_3 = 0
    if trbt_is_None == False:
        gamma_tilde = np.zeros((Nmode, Nmode, Nmode, Rthc, Rthc, Rthc))
        for (i,j,k), Gijk in gamma.items():
            u_max, v_max, w_max = Gijk.shape
            gamma_tilde[i,j,k,:u_max, :v_max, :w_max] = Gijk
        gamma_tilde = _symmetrize_gamma(gamma_tilde)
        lambda_3 += (1/8)*np.sum(np.abs(gamma_tilde))
        zeta_tilde += (3/2)*np.einsum('ijkUVW -> ijUV', gamma_tilde)

    # Factor of 1/4 below comes from converting number operator to reflection
    lambda_2 = (1/4)*np.sum(np.abs(zeta_tilde))    

    return lambda_1, lambda_2, lambda_3





@partial(jax.jit, static_argnums=(5,6))
def _hthc_vib_one_norm(kappa, tbt_full, trbt_full, zeta, gamma, obt_is_none, trbt_is_None):
    #This function assumes the input zeta is an array obtained by padding zeros to the original zeta dictionary
    lambda_1 = 0.0
    sym_tbt_full = (tbt_full + jnp.transpose(tbt_full, (3, 4, 5, 0, 1, 2)))/2
    if obt_is_none == False:
        kappa = kappa + (1)*jnp.einsum('ipqjrr -> ipq', sym_tbt_full)
        if trbt_is_None == False:
            kappa += (3/4)*jnp.einsum('ipqjrrktt -> ipq', trbt_full)
        Nmode = kappa.shape[0]
        for i in range (Nmode):                                 #Diagonalize obt in each mode
            kappa_i = kappa[i]
            D = jnp.linalg.eigvalsh(kappa_i)
            lambda_1 += (1/2)*jnp.sum(jnp.abs(D))

    zeta_tilde = zeta
    if trbt_is_None == False:
        zeta_tilde += (3/2)*jnp.einsum('ijkUVW -> ijUV', gamma)

    # Factor of 1/4 below comes from converting number operator to reflection. Note, this zeta is unsymmetrized but does not change the one-norm expression
    lambda_2 = (1/4)*jnp.sum(jnp.abs(zeta_tilde))


    lambda_3 = 0
    if trbt_is_None == False:
        lambda_3 += (1/8)*jnp.sum(jnp.abs(gamma))

    return lambda_1, lambda_2, lambda_3




@partial(jax.jit, static_argnums=(1,2,3,4,5,6,7))
def _extract_from_x_vec(x_vec, theta_size, Rthc, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None, indices):
    theta_1d, zeta_1d = jnp.split(x_vec, [theta_size])

    # Initialize full arrays of zeros
    theta = jnp.zeros((Nmode, Nmodal-1, Rthc))
    zeta = jnp.zeros((Nmode, Nmode, Rthc, Rthc))
    theta_indices, zeta_indices = indices
    
    # Scatter the 1D values into their specific structured locations
    theta = theta.at[theta_indices].set(theta_1d)
    zeta = zeta.at[zeta_indices].set(zeta_1d)

    if trbt_is_None == False:
        G_fin = Z_fin + Nmode**3*Nthc**3
        gamma = x_vec[Z_fin:G_fin].reshape((Nmode, Nmode, Nmode, Nthc, Nthc, Nthc))
        gamma = _symmetrize_gamma(gamma)
        Z_fin = G_fin
    else:
        gamma = None           

    if include_bliss:
        avec_len, bvec_len, ob_mat_num_params, dvec_len = _get_BLISS_sizes(num_ob_syms, Nmodal)
        a_fin = Z_fin + avec_len
        avec = x_vec[Z_fin:a_fin]

        b_fin = a_fin + bvec_len
        bvec = x_vec[a_fin:b_fin]

        beta_mats_fin = b_fin + ob_mat_num_params*num_ob_syms
        beta_mats_params = x_vec[b_fin:beta_mats_fin].reshape((num_ob_syms, ob_mat_num_params))

        dvec = x_vec[beta_mats_fin:]
    else:
        avec, bvec, beta_mats_params, dvec = None, None, None, None

    return theta, zeta, gamma, avec, bvec, beta_mats_params, dvec




@partial(jax.jit, static_argnums=(6,7,8,9,10,11,12,13,16,17))
def _cost_vib(x_vec, obt_full, tbt_full, trbt_full, ob_sym_mats, ob_sym_vals, theta_size, Rthc, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none, trbt_is_None, rho, indices, regularize=True, fro=True, norm_tbt_sq=0.0):
    theta, zeta, gamma, avec, bvec, beta_mats_params, dvec = _extract_from_x_vec(x_vec, theta_size, Rthc, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None, indices)
    
    # # --- REPLACED _tbt_error WITH _tbt_error_opt ---
    tot_cost = _tbt_error_opt(theta, zeta, tbt_full, norm_tbt_sq)

    # tot_cost = _tbt_error_stable_opt(theta, zeta, tbt_full)
    
    # if include_bliss:
    #     obt_killer, tbt_killer = _BLISS_corrections(avec, bvec, beta_mats_params, dvec, Nmodal, ob_sym_mats, ob_sym_vals, num_ob_syms)
    #     tbt_BI = tbt_full - tbt_killer
    #     if obt_is_none:
    #         obt_tilde = None
    #     else:
    #         obt_tilde = obt_full - obt_killer + jnp.einsum("ipqjrr->ipq", tbt_BI)
    # else:
    #     tbt_BI = tbt_full
    #     if obt_is_none:
    #         obt_tilde = None
    #     else:
    #         obt_tilde = obt_full - (1/2)*jnp.einsum('ipqjrr -> ipq', tbt_full)
    #         if trbt_is_None == False:
    #             obt_tilde -= (3/8)*jnp.einsum('ipqjrrktt -> ipq', trbt_full)


    if trbt_is_None == False:
        trbt_diff = trbt_full - trbt_thc
        tot_cost += ten_norm(trbt_diff, fro=fro)

    if regularize:
        one_norm = sum(_hthc_vib_one_norm(obt_full, tbt_full, trbt_full, zeta, gamma, obt_is_none, trbt_is_None))
        # print (jax.debug.print('One norm  = {}', one_norm))
        tot_cost += rho * one_norm
    
    return tot_cost







def get_vib_hthc(tbt, trbt=None, obt=None, ob_sym_list=[], Nthc=None, regularize=True, maxiter=10000, initial_guess=None, learning_rate = 7.5e-3, verbose=True, fro=True, chunk_size = 200):
    """
    Function to perform heterogeneous THC i.e. number of THC orbitals can be different for different modes.
    Input:
    ------
    tbt: Two body tensor of shape (Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal)
    trbt: Three body tensor of shape (Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal)
    obt: One body tensor of shape (Nmode, Nmodal, Nmodal)
    ob_sym_list: List of symmetry operators (not implemented)
    Nthc: list of number of THC orbitals for each mode
    """

    Nmode = tbt.shape[0]
    Nmodal = tbt.shape[-1]

    if Nthc is None:
        Nthc = (int(np.ceil(Nmodal+1)),)*Nmode
        if verbose:
            print(f"Using default homogeneous THC rank of ceil(num_modals+1) = {Nthc}")
    elif type(Nthc) is int:
        Nthc = (Nthc,)*Nmode
        print (f"Performing Homogeneous THC: This function might be slover than the other implementation")
    elif type(Nthc) is list:
        if len(Nthc) != Nmode:
            raise ("Nthc must of length Nmode if not an integer")
        Nthc = tuple(Nthc)
    Rthc = max(Nthc)


    #___________________________________________________________________________________________________________________________________
    #___________________________________________Bliss symmetries and their coefficients_________________________________________________
    num_ob_syms = len(ob_sym_list)
    avec_len, bvec_len, ob_mat_num_params, dvec_len = _get_BLISS_sizes(num_ob_syms, Nmodal)
    beta_params_len = num_ob_syms * ob_mat_num_params

    if obt is None:
        obt_is_none = True
    else:
        obt_is_none = False

    if trbt is None:
        trbt_is_None = True
    else:
        trbt_is_None = False

    if num_ob_syms > 0:
        include_bliss = True
    else:
        include_bliss = False

    ob_sym_mats = jnp.array([ob_sym_list[kk][0] for kk in range(num_ob_syms)])
    ob_sym_vals = jnp.array([ob_sym_list[kk][1] for kk in range(num_ob_syms)])

    if verbose and include_bliss:
        print(f"Found {num_ob_syms} one-body symmetries for BLISS terms")
        print(f"Total number of BLISS parameters to be optimized: {avec_len+bvec_len+beta_params_len}, composed of:")
        print(f"    - {avec_len} one-body scalars")
        print(f"    - {bvec_len} two-body scalars")
        print(f"    - {num_ob_syms} one-body matrices, each with {ob_mat_num_params} free variables")
        print(f"    - {dvec_len} two-body scalars mixing beta matrices with symmetries")
    #___________________________________________________________________________________________________________________________________
    #___________________________________________________________________________________________________________________________________




    #___________________________________________________________________________________________________________________________________
    #_________________________________________________Setting up initial guess and rho__________________________________________________
    if initial_guess is None:
        theta={}
        for i in range(Nmode):
            theta_r = np.random.uniform(low=-np.pi, high=np.pi, size=(Nmodal - 1, Nthc[i]))
            theta[i] = theta_r
        zeta={}
        for i in range(Nmode):
            for j in range(i):
                zeta[(i,j)] = np.zeros((Nthc[i], Nthc[j]))
        params={}
        params['theta'] = theta
        params['zeta'] = zeta
        if trbt_is_None == False:
            gamma = {}
            for i in range(Nmode):
                for j in range(i):
                    for k in range (j):
                        gamma[(i,j,k)] = np.zeros((Nthc[i], Nthc[j], Nthc[k]))
        if verbose:
          print ("Initial guess is set to None")
    elif isinstance(initial_guess, dict):
        params = copy(initial_guess)
        if trbt_is_None == False:
            if 'gamma' not in params:
                params['gamma'] = np.zeros((Nmode, Nmode, Nmode, Nthc, Nthc, Nthc))
        if verbose:
          print ("Initial guess is user provided")
        

    if regularize is False or regularize is None:
        rho = 0
        if verbose:
            print(f"No regularization: setting rho=0")
    else:
        rho = regularize
        regularize=True
        if verbose:
            print(f"Regularization found: setting rho={rho:.2e}")
    #___________________________________________________________________________________________________________________________________
    #___________________________________________________________________________________________________________________________________

    def build_zeta_index_map(zeta_dict):
        """
        Creates a flat 1D array of zeta values and a tuple of indices
        mapping them to a padded array.
        """
        flat_values = []
        i_idx, j_idx, u_idx, v_idx = [], [], [], []
        
        for (i, j), Zij in zeta_dict.items():
            U, V = Zij.shape
            for u in range(U):
                for v in range(V):
                    flat_values.append(Zij[u, v])
                    i_idx.append(i)
                    j_idx.append(j)
                    u_idx.append(u)
                    v_idx.append(v)
                    
        # Convert to JAX arrays. 
        # The indices will be treated as static constants by the JIT compiler.
        return (jnp.array(flat_values), 
                (jnp.array(i_idx), jnp.array(j_idx), jnp.array(u_idx), jnp.array(v_idx)))

    
    def build_theta_index_map(theta):
        """
        Creates a flat 1D array of theta values and a tuple of indices
        mapping them to a padded array.
        """
        flat_values = []
        i_idx, p_idx, u_idx = [], [], []

        for i, theta_i in theta.items():
            P, U = theta_i.shape
            for p in range(P):
                for u in range(U):
                    flat_values.append(theta_i[p, u])
                    i_idx.append(i)
                    p_idx.append(p)
                    u_idx.append(u)

        # Convert to JAX arrays.
        # The indices will be treated as static constants by the JIT compiler.
        return (jnp.array(flat_values),
                (jnp.array(i_idx), jnp.array(p_idx), jnp.array(u_idx)))

    zeta_vec, zeta_indices = build_zeta_index_map(params['zeta'])
    theta_vec, theta_indices = build_theta_index_map(params['theta'])
    x0 = jnp.concatenate([theta_vec, zeta_vec])
    theta_size = theta_vec.shape[0]
    indices = (theta_indices, zeta_indices)


    def pack_2dict(theta, zeta, Nthc, Nmode, gamma=None, avec=None, bvec=None, beta_mats_params=None, dvec=None):
        theta_c = {i: np.array(theta[i, :, :Nthc[i]]) for i in range(Nmode)}
        zeta_c = {(i,j): np.array(zeta[i, j, :Nthc[i], :Nthc[j]]) for i in range (Nmode) for j in range(i)}
        my_dict = {"theta" : theta_c, "zeta" : zeta_c, "Nthc": Nthc}
        if trbt_is_None == False:
            my_dict["gamma"] = gamma
        if include_bliss:
            my_dict["avec"] = avec
            my_dict["bvec"] = bvec
            my_dict["beta_mats_params"] = beta_mats_params
            my_dict["dvec"] = dvec
        return my_dict


    jnp_tbt = jnp.array(tbt)

    # --- NEW: Precompute the constant target norm ---
    norm_tbt_sq = jnp.sum(jnp_tbt**2)

    @jit
    def cost_flat(x_vec):
        return _cost_vib(x_vec, obt, jnp_tbt, trbt, ob_sym_mats, ob_sym_vals, theta_size, Rthc, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none=obt_is_none, trbt_is_None=trbt_is_None, rho=rho, indices=indices, regularize=regularize, fro=fro, norm_tbt_sq=norm_tbt_sq)
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(x0)

    # --- NEW: Compile chunk_size iterations into a single XLA execution ---
    
    @jit
    def update_chunk(x_vec, opt_state):
        def scan_step(carry, _):
            x, state = carry
            loss, grads = jax.value_and_grad(cost_flat)(x)
            updates, state = optimizer.update(grads, state, x)
            x = optax.apply_updates(x, updates)
            return (x, state), loss

        # Run the step chunk_size times entirely on the backend
        (final_x, final_state), loss_history = jax.lax.scan(
            scan_step, 
            (x_vec, opt_state), 
            None, 
            length=chunk_size
        )
        return final_x, final_state, loss_history
    # ---------------------------------------------------------------

    # Optimization loop setup
    losses = []
    org_diff = ten_norm(tbt, fro=fro)
    if trbt_is_None == False:
        org_diff += ten_norm(trbt, fro=fro)
    print ("Original tensor norm: ", org_diff, flush = True)
    
    int_theta, int_zeta, int_gamma, _, _, _, _ = _extract_from_x_vec(x0, theta_size, Rthc, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None, indices)
    
    # Calculate how many chunk_size-step blocks to run
    num_chunks = maxiter // chunk_size

    for chunk in range(num_chunks):
        # i maps to 0, 100, 200, 300, etc. if chunk_size = 100
        i = chunk * chunk_size
        
        # 1. Evaluate and print intermediate state BEFORE the chunk runs
        int_theta, int_zeta, int_gamma, _, _, _, _ = _extract_from_x_vec(x0, theta_size, Rthc, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None, indices)
        int_norm = sum(_hthc_vib_one_norm(obt, jnp_tbt, trbt, int_zeta, int_gamma, obt_is_none, trbt_is_None))
        int_error = _tbt_error_opt(int_theta, int_zeta, jnp_tbt, norm_tbt_sq)
        if trbt_is_None == False:
            int_error += ten_norm(int_trbt - trbt, fro=fro)

        print (f'Iter {i}: (one-norm, error) = ({int_norm}, {int_error.item()})', flush = True)

        # 2. Execute chunk_size steps
        x0, opt_state, chunk_losses = update_chunk(x0, opt_state)
        
        # 3. Pull the chunk_size losses back to Python memory once
        losses.extend(np.array(chunk_losses).tolist())

        if verbose == True and i % 1000 == 0:
            # Print the very last loss of the current chunk
            print(f"Iteration {i}: Loss = {losses[-1]:.6e}", flush = True)
        
        # # Simple convergence check (using the last two elements of the entire history)
        # if len(losses) > 1 and abs(losses[-1] - losses[-2]) < 1e-12:
        #     if verbose == True:
        #         print(f"Converged around iteration {i + chunk_size}", flush = True)
        #     break
        
    theta, zeta, gamma, avec, bvec, beta_mats_params, dvec = _extract_from_x_vec(x0, theta_size, Rthc, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None, indices)    
    final_params = pack_2dict(theta, zeta, Nthc, Nmode, gamma, avec, bvec, beta_mats_params, dvec)
    L2_cost = _cost_vib(x0, obt, jnp_tbt, trbt, ob_sym_mats, ob_sym_vals, theta_size, Rthc, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none, trbt_is_None, 0, indices, False, fro, norm_tbt_sq)
    lam = float(sum(_hthc_vib_one_norm(obt, jnp_tbt, trbt, zeta, gamma, obt_is_none, trbt_is_None)))

    if verbose:
        print(f"\nInitial norm is {float(ten_norm(tbt, fro=fro)):.2e}")
        print(f"Finished THC factorization! Final norm of difference is {L2_cost:.2e}, 1-norm is {lam:.2f}\n")
        if obt_is_none:
            print(f"Note that one-norm does not include one-body component!")

        if include_bliss:
            print(f"BLISS included during optimization using {num_ob_syms} one-body symmetries")

    return final_params, lam