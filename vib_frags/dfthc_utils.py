import numpy as np
import jax
from jax import numpy as jnp
from jax import grad, jit, vmap
from jax import lax
import jax.scipy.optimize
import optax
import time
import pybtas
from functools import partial
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from copy import copy

tl.set_backend('numpy')   # use numpy backend (or 'pytorch','jax' if desired)

def cp_then_symmetrize(T, N, rank, parafac_kwargs=None):
    """
    G: (N*N, R) -> tensor T of shape (N,N,R)
    rank: CP rank (number of components)
    returns: A (N, rank), B (R, rank)
    """
    if parafac_kwargs is None:
        parafac_kwargs = dict(n_iter_max=20000, tol=1e-6, init='svd', verbose=False)

    # T = G.reshape((N, N, G.shape[1]))  # shape (N, N, R)

    # run standard CP (factors is a list of factor matrices [A0, A1, B])
    weights, factors = parafac(T, rank=rank, **parafac_kwargs, return_errors=False)

    A0, A1, B = factors  # shapes: (N,rank), (N,rank), (R,rank)
    # absorb weights into B (tensorly returns weights vector)
    B = B * weights[np.newaxis, :]

    # symmetrize A0 and A1 by averaging their columns
    A_sym = 0.5 * (A0 + A1)

    # renormalize columns of A_sym and absorb scale into B (A appears twice)
    col_norms = np.linalg.norm(A_sym, axis=0)
    col_norms_safe = np.where(col_norms == 0, 1.0, col_norms)
    A_sym = A_sym / col_norms_safe[np.newaxis, :]
    B = B * (col_norms_safe[np.newaxis, :] ** 2)

    return A_sym, B

def _get_BLISS_sizes(num_ob_syms, Norbs):
    avec_len = num_ob_syms
    bvec_len = int(num_ob_syms * (num_ob_syms+1)/2)
    ob_mat_num_params = int(Norbs*(Norbs+1)/2)
    dvec_len = int(num_ob_syms * (num_ob_syms-1)/2)

    return avec_len, bvec_len, ob_mat_num_params, dvec_len



def _verify_vib_tbt_symmetries(tbt):
    print ("Still needs to be implemented")
    raise NotImplementedError


def _symmetrize_gamma(gamma):
    gamma += np.transpose(gamma, (0,2,1,3,5,4))
    gamma += np.transpose(gamma, (1,0,2,4,3,5))
    gamma += np.transpose(gamma, (1,2,0,4,5,3))
    gamma += np.transpose(gamma, (2,0,1,5,3,4))
    gamma += np.transpose(gamma, (2,1,0,5,4,3))
    return gamma*(1/2**5)

def _get_vib_cp3(tbt, Nthc, fro=True, verify=False, first_factor_thresh=1.0e-14, random_start_thc=True, conv_eps=1.0e-4, verbose=False):
    "Assume tbt is of shape (Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal)"
    Nmode = tbt.shape[0]
    Nmodal = tbt.shape[-1]
    # if verify:
        # _verify_vib_tbt_symmetries(tbt)

    # Perform eigen decomposition
    tbt_mat = tbt.reshape((Nmode*Nmodal**2, Nmode*Nmodal**2))
    w, u = np.linalg.eigh(tbt_mat)

    # Get non-zero eigenvalues
    non_zero_w = np.where(np.abs(w) >= first_factor_thresh)[0]
    # Get index of positive and negative eigenvalues and prepare for CP3
    w_signs = (w[non_zero_w] >= 0).astype(int) * 2 - 1

    u_chol = u[:, non_zero_w] @ np.diag(np.sqrt(np.abs(w[non_zero_w])))
    u_chol_reshaped = u_chol.reshape((Nmode, Nmodal**2, len(non_zero_w)))
    
    if verify:
        assert np.allclose(tbt_mat, tbt_mat.T)
        assert np.allclose(u @ np.diag(w) @ u.T, tbt_mat)

    # CP3 decomposition per each mode
    thc_leaves = np.zeros((Nmode, Nthc, Nmodal))
    thc_centrals = np.zeros((Nmode, Nmode, Nthc, Nthc))
    thc_gammas = np.zeros((Nmode, Nthc, len(non_zero_w)))
    for i in range (Nmode):
        u_chol_i = u_chol_reshaped[i]

        start_time = time.time()
        beta, gamma, scale = pybtas.cp3_from_cholesky(u_chol_i.copy(), Nthc, random_start=random_start_thc, conv_eps=conv_eps)
        cp3_calc_time = time.time() - start_time

        thc_leaf = beta.T
        gamma = gamma.T

        thc_gamma = np.einsum('ur,u->ur', gamma, scale.ravel())

        thc_leaves[i] = thc_leaf
        thc_gammas[i] = thc_gamma
    

    # Generate THC central for each pair of modes (i,j)
    for i in range (Nmode):
        for j in range (Nmode):
            if i != j:
                thc_centrals[i, j] = np.einsum('ur, vr, r -> uv', thc_gammas[i], thc_gammas[j], w_signs)

    if verify:
        print("\tCP3 timing: ", cp3_calc_time)

    if verify:
        assert np.allclose(thc_centrals, np.transpose(thc_centrals, (1,0,3,2)))
    
    #Check quality of initial guess
    CiUpq = np.einsum("iUp,iUq->iUpq", thc_leaves, thc_leaves)
    tbt_cp3 = np.einsum('ijUV,iUpq,jVrs->ipqjrs', thc_centrals, CiUpq, CiUpq) 
    print ("Error after initial guess from CP3: ", ten_norm(tbt - tbt_cp3, fro=fro))
    return thc_centrals, thc_leaves



# @jit
# partial(jax.jit, static_argnums=(3))  # no static args required, but jittable
def unfold_vib_thc(theta, sigma, phi, gamma, trbt_is_None):
    Nmode = theta.shape[0]
    Nthc = theta.shape[1]

    xi = theta_to_normalized_xi(theta)
    CiUpq = jnp.einsum("iUp,iUq->iUpq", xi, xi)
    
    V = theta_to_normalized_xi(phi).T
    zeta = jnp.einsum('xa,a,ya-> xy', V, sigma, V)
    zeta = zeta.reshape((Nmode, Nthc, Nmode, Nthc))
    zeta = jnp.transpose(zeta, (0,2,1,3))
    
    tbt_thc = jnp.einsum('ijUV,iUpq,jVrs->ipqjrs', zeta, CiUpq, CiUpq)
    if not trbt_is_None:
        trbt_thc = jnp.einsum('ijkUVW,iUpq,jVrs,kWtu->ipqjrsktu', gamma, CiUpq, CiUpq, CiUpq)
    else:
        trbt_thc = None
    return tbt_thc, trbt_thc
unfold_vib_thc = jit(unfold_vib_thc, static_argnums=(4))



def ten_norm(ten, fro=True):
    """
    Input tensor can be numpy or jaxnumpy. If fro=True, L2 norm of the flatenned array (Frobenius norm) is returned, 
    otherwise L1 norm of the flattened array is returned.
    """
    if fro == True:
        return jnp.sqrt(jnp.sum(ten**2))
    else:
        return jnp.sum(jnp.abs(ten))




def _compute_vib_penalty_param(tbt, trbt, obt, ob_sym_list, Nthc, initial_guess, maxiter, improve_guess=False):
    """Compute penalty parameter by using un-regularized two-norm of difference
    Will do maxiter iterations of optax optimizer if improve_guess is set as True"""
    if improve_guess:
        print(f"Running initial conditions through {maxiter} iterations for computing penalty parameter...")
        params, _ = get_vib_thc(tbt, obt, ob_sym_list, Nthc, initial_guess=initial_guess, regularize=False, maxiter=maxiter, verbose=False)
    else:
        params = initial_guess
    
    xi = params['xi']
    zeta = params['zeta']
    
    dtbt = tbt - unfold_vib_thc(zeta, xi)
    sum_square_loss = 0.5 * jnp.sum(dtbt**2)
    regularization_scale = jnp.sum(jnp.abs(zeta))
    
    # Avoid division by zero
    if regularization_scale < 1e-12:
        return 1e-6
    
    # return float(sum_square_loss / regularization_scale), params
    return float(5e-5), params




@partial(jax.jit, static_argnums=())  # no static args required
def theta_to_normalized_xi(params: jnp.ndarray) -> jnp.ndarray:
    """
    params: array shape (nmodes, nthc, nmodals-1)
    returns: array shape (nmodes, nthc, nmodals) -- last axis is the unit vector
    """
    # if params.ndim != 3:
    #     raise ValueError("params must have shape (nmodes, nthc, nmodals-1)")

    # trig
    sin_th = jnp.sin(params)   # shape (..., K-1)
    cos_th = jnp.cos(params)   # shape (..., K-1)

    # prefix products along last axis: prod_{i=0..k} sin_th[..., i]
    prefix = jnp.cumprod(sin_th, axis=-1)  # shape (..., K-1)

    # factors: factor[..., 0] = 1, factor[..., k] = prefix[..., k-1] for k>=1
    # create an array of same shape as prefix to hold factors for first K-1 components
    ones = jnp.ones(prefix.shape[:-1] + (1,), dtype=prefix.dtype)  # (..., 1)
    if prefix.shape[-1] == 1:
        factors = ones  # only factor for k=0 (since K-1 == 1)
    else:
        # prefix[..., :-1] has shape (..., K-2). Concatenate leading ones to make (..., K-1)
        factors = jnp.concatenate([ones, prefix[..., :-1]], axis=-1)  # (..., K-1)

    # first K-1 components
    comps_first = factors * cos_th  # (..., K-1)

    # last component is product of all sines -> prefix[..., -1]
    last_comp = prefix[..., -1]  # shape (...)

    # stack along last axis to get (..., K)
    last_comp_expanded = jnp.expand_dims(last_comp, axis=-1)  # (..., 1)
    result = jnp.concatenate([comps_first, last_comp_expanded], axis=-1)  # (..., K)

    return result





@partial(jax.jit, static_argnums=())
def normalized_xi_to_theta(v: jnp.ndarray, make_last_in_0_2pi: bool = True) -> jnp.ndarray:
    """
    Inverse hyperspherical mapping (batched).
    Inputs
    ------
    v : jnp.ndarray of shape (..., N)  (e.g. (nmodes, nthc, nmodals))
        Each last-axis vector is assumed normalized (||v|| ~= 1).
    make_last_in_0_2pi : bool, optional
        If True, map the last angle to [0, 2*pi). If False, last angle is in (-pi, pi].

    Returns
    -------
    thetas : jnp.ndarray of shape (..., N-1)
        Angles theta_1 .. theta_{N-1} corresponding to each vector.
        Ranges: theta_1..theta_{N-2} in [0, pi], theta_{N-1} as described above.
    """
    if v.ndim < 1:
        raise ValueError("v must have at least 1 dimension and last axis length >= 2")
    N = v.shape[-1]
    if N < 2:
        raise ValueError("last axis length must be at least 2 (N>=2)")

    # compute squared tail norms: tail_sq[k] = sum_{m=k..N-1} v[..., m]^2 in 0-based indexing
    # We'll produce r_k = sqrt(sum_{m=k..N-1} v[..., m]^2) in 0-based indexing where k in [0..N-1].
    # Convenient approach: compute cumulative sums from the end:
    sq = jnp.square(v)  # (..., N)
    # cumulative sum from the right: rev_cumsum = cumsum(rev(sq)) then reverse back
    rev_sq = jnp.flip(sq, axis=-1)
    rev_cumsum = jnp.cumsum(rev_sq, axis=-1)
    tail_sq = jnp.flip(rev_cumsum, axis=-1)  # (..., N), tail_sq[..., k] = sum_{m=k..N-1} sq[..., m]
    # numerical safety: clip to non-negative
    tail_sq = jnp.clip(tail_sq, a_min=0.0)

    r = jnp.sqrt(tail_sq)  # (..., N), r[..., k] = sqrt(sum_{m=k..N-1} v[..., m]^2)
    # r[..., 0] should be ~1 for normalized vectors

    # For theta_k (k=1..N-2) in 1-based math indexing, corresponds to k_idx = 0..N-3 in 0-based:
    # theta_k = atan2(r_{k+1}, x_k) where r_{k+1} corresponds to r[..., k+1].
    if N == 2:
        # special-case: only one angle theta_1 = atan2(x2, x1)
        theta_last = jnp.arctan2(v[..., 1], v[..., 0])
        if make_last_in_0_2pi:
            theta_last = jnp.where(theta_last < 0.0, theta_last + 2.0 * jnp.pi, theta_last)
        return theta_last[..., jnp.newaxis]  # shape (..., 1)

    # compute the first N-2 theta:
    # k_idx runs 0 .. N-3
    x_k = v[..., : (N - 2)]          # (..., N-2) when N>=3
    r_kplus1 = r[..., 1 : (N - 1)]   # (..., N-2), r_{k+1} for k=0..N-3
    thetas_first = jnp.arctan2(r_kplus1, x_k)  # (..., N-2), in [0, pi]

    # last angle (theta_{N-1}): use atan2(x_N, x_{N-1})
    theta_last = jnp.arctan2(v[..., -1], v[..., -2])  # shape (...)

    if make_last_in_0_2pi:
        theta_last = jnp.where(theta_last < 0.0, theta_last + 2.0 * jnp.pi, theta_last)

    # now concatenate:
    thetas = jnp.concatenate([thetas_first, theta_last[..., jnp.newaxis]], axis=-1)  # (..., N-1)
    return thetas





def _extract_from_x_vec(x_vec, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None):
        theta_fin = Nmode*Nthc*(Nmodal-1)
        theta = x_vec[:theta_fin].reshape((Nmode, Nthc, Nmodal-1))

        sigma_fin = theta_fin + Nsig
        sigma = x_vec[theta_fin:sigma_fin]

        phi_fin = sigma_fin + Nsig*(Nmode*Nthc-1)
        phi = x_vec[sigma_fin:phi_fin].reshape((Nsig, Nmode*Nthc-1))

        if trbt_is_None == False:
            G_fin = phi_fin + Nmode**3*Nthc**3
            gamma = x_vec[phi_fin:G_fin].reshape((Nmode, Nmode, Nmode, Nthc, Nthc, Nthc))
            gamma = _symmetrize_gamma(gamma)
            phi_fin = G_fin
        else:
            gamma = None           

        if include_bliss:
            avec_len, bvec_len, ob_mat_num_params, dvec_len = _get_BLISS_sizes(num_ob_syms, Nmodal)
            a_fin = phi_fin + avec_len
            avec = x_vec[phi_fin:a_fin]

            b_fin = a_fin + bvec_len
            bvec = x_vec[a_fin:b_fin]

            beta_mats_fin = b_fin + ob_mat_num_params*num_ob_syms
            beta_mats_params = x_vec[b_fin:beta_mats_fin].reshape((num_ob_syms, ob_mat_num_params))

            dvec = x_vec[beta_mats_fin:]
        else:
            avec, bvec, beta_mats_params, dvec = None, None, None, None

        return theta, sigma, phi, gamma, avec, bvec, beta_mats_params, dvec

_extract_from_x_vec = jit(_extract_from_x_vec, static_argnums=(1,2,3,4,5,6,7))




def _thc_vib_one_norm(kappa, tbt_full, trbt_full, sigma, phi, gamma, obt_is_none, trbt_is_None):
    lambda_1 = 0.0
    if obt_is_none == False:
        kappa = kappa - (1)*jnp.einsum('ipqjrr -> ipq', tbt_full)
        if trbt_is_None == False:
            kappa -= (3/4)*jnp.einsum('ipqjrrktt -> ipq', trbt_full)
        Nmode = kappa.shape[0]
        for i in range (Nmode):                                 #Diagonalize obt in each mode
            kappa_i = kappa[i]
            D = jnp.linalg.eigvalsh(kappa_i)
            lambda_1 += (1/2)*jnp.sum(jnp.abs(D))

    V = theta_to_normalized_xi(phi).T
    zeta = jnp.einsum('xa,a,ya-> xy', V, sigma, V)

    zeta_tilde = zeta
    if trbt_is_None == False:
        zeta_tilde += (3/2)*jnp.einsum('ijkUVW -> ijUV', gamma)
    lambda_2 = (1/4)*jnp.sum(jnp.abs(zeta_tilde))

    lambda_3 = 0
    if trbt_is_None == False:
        lambda_3 += (1/8)*jnp.sum(jnp.abs(gamma))

    return lambda_1, lambda_2, lambda_3

_thc_vib_one_norm = jit(_thc_vib_one_norm, static_argnums=(6,7))




def _cost_vib(x_vec, obt_full, tbt_full, trbt_full, ob_sym_mats, ob_sym_vals, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none, trbt_is_None, rho, regularize=True, fro=True):
    theta, sigma, phi, gamma, avec, bvec, beta_mats_params, dvec = _extract_from_x_vec(x_vec, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None)
    tbt_thc, trbt_thc = unfold_vib_thc(theta, sigma, phi, gamma, trbt_is_None)
    
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

    tbt_diff = tbt_full - tbt_thc
    tot_cost = ten_norm(tbt_diff, fro=fro)

    if trbt_is_None == False:
        tbt_diff = trbt_full - trbt_thc
        tot_cost += ten_norm(tbt_diff, fro=fro)

    if regularize:
        one_norm = sum(_thc_vib_one_norm(obt_full, tbt_full, trbt_full, sigma, phi, gamma, obt_is_none, trbt_is_None))
        # print (jax.debug.print('One norm  = {}', one_norm))
        tot_cost += rho * one_norm
    
    return tot_cost

_cost_vib = jit(_cost_vib, static_argnums=(6,7,8,9,10,11,12,13,15,16))





def _vib_vec_to_one_norm(x_vec, obt_full, tbt_full, trbt_full, ob_sym_mats, ob_sym_vals, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none, trbt_is_None):
    theta, sigma, phi, gamma, avec, bvec, beta_mats_params, dvec = _extract_from_x_vec(x_vec, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None)

    # if include_bliss:
    #     obt_killer, tbt_killer = _BLISS_corrections(avec, bvec, beta_mats_params, dvec, Nmodal, ob_sym_mats, ob_sym_vals, num_ob_syms)
    #     tbt_BI = tbt_full - tbt_killer
    #     if obt_is_none:
    #         obt_BI = None
    #     else:
    #         obt_BI = obt_full - obt_killer
    # else:
    #     tbt_BI = tbt_full
    #     obt_BI = obt_full

    # if obt_is_none:
    #     kappa_BI = None
    # else:
    #     kappa_BI = obt_BI - (1/2)*jnp.einsum('ipqjrr -> ipq', tbt_BI)
    #     if trbt_is_None == False:
    #         obt_tilde -= (3/8)*jnp.einsum('ipqjrrktt -> ipq', trbt_full)

    return sum(_thc_vib_one_norm(obt_full, tbt_full, trbt_full, sigma, phi, gamma, obt_is_none, trbt_is_None))

_vib_vec_to_one_norm = jit(_vib_vec_to_one_norm, static_argnums=(6,7,8,9,10,11,12,13))




def get_vib_dfthc_v1(tbt, trbt=None, obt=None, ob_sym_list=[], Nthc=None, Nsig=None, regularize=True, maxiter=10000, initial_guess=None, learning_rate = 7.5e-3, verbose=True, fro=True):
    """
    Input:
        tbt: two body tensor of shape (Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal)
        trbt: three body tensor of shape (Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal)
        obt: one body tensor of shape (Nmode, Nmodal, Nmodal)
        Nthc: Number of THC orbitals
        Nsig: Number of singular values of the THC Zeta matrix
    """
    Nmode = tbt.shape[0]
    Nmodal = tbt.shape[-1]

    if Nthc is None:
        Nthc = Nmodal + 2
        if verbose:
            print(f"Using default THC rank of num_modals + 2 = {Nthc}")
    
    if Nsig is None:
        Nsig = int(np.ceil(Nthc/2))
        if verbose:
            print(f"Using default Zeta matrix rank of Nthc/2 = {Nsig}")

    #Bliss symmetries and their coefficients
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

    #params = _initialize_vib_params(initial_guess, tbt, Nthc, Nmodal, include_bliss, num_ob_syms)

    if initial_guess == None:
        params={}
        params['theta'] = np.random.uniform(low=0, high=2*np.pi, size=(Nmode, Nthc, Nmodal - 1))
        params['sigma'] = np.ones(Nsig)
        params['phi'] = np.random.uniform(low=0, high=2*np.pi, size=(Nsig, Nmode*Nthc-1))

        if trbt_is_None == False:
            params['gamma'] = np.zeros((Nmode, Nmode, Nmode, Nthc, Nthc, Nthc))
        print ("Initial guess is set to None")
    elif initial_guess == 'cp3':
        zeta, xi = _get_vib_cp3(tbt, Nthc, fro)
        zeta = np.transpose(zeta, (0,2,1,3))
        zeta_mat = zeta.reshape((Nmode*Nthc, Nmode*Nthc))
        sigmas, eig_vecs = np.linalg.eigh(zeta_mat)
        sigmas_srtd = sigmas[np.argsort(np.abs(sigmas))[:Nsig]]
        eig_vecs_srtd = eig_vecs[:, np.argsort(np.abs(sigmas))[:Nsig]]
        phi = normalized_xi_to_theta(eig_vecs_srtd.T)

        theta = normalized_xi_to_theta(xi)
        params={}
        params['theta'] = theta
        params['sigma'] = sigmas_srtd
        params['phi'] = phi
        if trbt_is_None == False:
            params['gamma'] = np.zeros((Nmode, Nmode, Nmode, Nthc, Nthc, Nthc))
        print ("Initial guess is set to cp3")
    elif isinstance(initial_guess, dict):
        params = copy(initial_guess)
        if len(params['sigma']) < Nsig:
            sigmas = params['sigma']
            params['sigma'] = np.concatenate((sigmas, np.zeros(Nsig-len(sigmas))))
            params['phi'] = np.vstack((params['phi'], np.zeros((Nsig-len(sigmas), Nmode*Nthc-1))))
        if trbt_is_None == False:
            if 'gamma' not in params:
                params['gamma'] = np.zeros((Nmode, Nmode, Nmode, Nthc, Nthc, Nthc))
        print ("Initial guess is user provided")
        


    if regularize is True:
        rho, params = _compute_vib_penalty_param(tbt, obt, ob_sym_list, Nthc, params, int(np.ceil(maxiter/10)))
        # if include_bliss:
        #     rho *= 2 / Nmode

        if verbose:
            print(f"Found regularization parameter rho={rho:.2e}")
    elif regularize is False or regularize is None:
        rho = 0
        if verbose:
            print(f"No regularization: setting rho=0")
    else:
        rho = regularize
        regularize=True
        if verbose:
            print(f"Regularization found: setting rho={rho:.2e}")

    def pack_2dict(x_vec):
        theta, sigma, phi, gamma, avec, bvec, beta_mats_params, dvec = _extract_from_x_vec(x_vec, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None)    
        my_dict = {"theta" : theta, "sigma" : sigma, "phi": phi}
        if trbt_is_None == False:
            my_dict["gamma"] = gamma
        if include_bliss:
            my_dict["avec"] = avec
            my_dict["bvec"] = bvec
            my_dict["beta_mats_params"] = beta_mats_params
            my_dict["dvec"] = dvec
        return my_dict

    def unpack_dict(my_dict):
        num_vars = Nmode * Nthc * (Nmodal-1) + Nsig + Nsig * (Nmode*Nthc-1)
        
        if trbt_is_None == False:
            num_vars += Nmode**3 * Nthc**3

        if include_bliss:
            num_vars += avec_len + bvec_len + beta_params_len + dvec_len

        x_vec = np.zeros(num_vars)

        theta_fin = Nmode*Nthc*(Nmodal-1)                                    
        x_vec[:theta_fin] = my_dict["theta"].flatten()

        sig_fin = theta_fin + Nsig
        x_vec[theta_fin:sig_fin] = my_dict["sigma"].flatten()

        phi_fin = sig_fin + Nsig*(Nmode*Nthc-1)
        x_vec[sig_fin:phi_fin] = my_dict["phi"].flatten()


        if trbt_is_None == False:
            G_fin = sig_fin + Nmode**3*Nthc**3
            x_vec[sig_fin:G_fin] = my_dict["gamma"].flatten()
            sig_fin = G_fin

        if include_bliss:
            a_fin = sig_fin + avec_len
            x_vec[sig_fin:a_fin] = my_dict["avec"]

            b_fin = a_fin + bvec_len
            x_vec[a_fin:b_fin] = my_dict["bvec"]

            beta_fin = b_fin + num_ob_syms*ob_mat_num_params
            x_vec[b_fin:beta_fin] = my_dict["beta_mats_params"].flatten()

            x_vec[beta_fin:] = my_dict["dvec"]

        return jnp.array(x_vec)

    @jit
    def cost_flat(x_vec):
        return _cost_vib(x_vec, obt, tbt, trbt, ob_sym_mats, ob_sym_vals, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none=obt_is_none, trbt_is_None=trbt_is_None, rho=rho, regularize=regularize)

    optimizer = optax.adam(learning_rate)
    x0 = unpack_dict(params)
    opt_state = optimizer.init(x0)

    @jit
    def update_step(x_vec, opt_state):
        loss, grads = jax.value_and_grad(cost_flat)(x_vec)
        updates, opt_state = optimizer.update(grads, opt_state, x_vec)
        x_vec = optax.apply_updates(x_vec, updates)
        return x_vec, opt_state, loss

    # Optimization loop
    losses = []
    org_diff = ten_norm(tbt, fro=fro)
    if trbt_is_None == False:
        org_diff += ten_norm(trbt, fro=fro)
    print ("Original tensor norm: ", org_diff)
    # int_theta, int_sigma, int_phi, int_gamma, _, _, _, _ = _extract_from_x_vec(x0, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None)
    for i in range(maxiter):
        #Check intermediate one-norm
        if  i % 100 == 0:
            # int_theta_c, int_sigma_c, int_phi_c = copy(int_theta), copy(int_sigma), copy(int_phi)
            int_norm = _vib_vec_to_one_norm(x0, obt, tbt, trbt, ob_sym_mats, ob_sym_vals, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none, trbt_is_None)
            int_theta, int_sigma, int_phi, int_gamma, _, _, _, _ = _extract_from_x_vec(x0, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None)
            int_tbt, int_trbt = unfold_vib_thc(int_theta, int_sigma, int_phi, int_gamma, trbt_is_None)
            int_error = ten_norm(int_tbt - tbt, fro=fro)
            if trbt_is_None == False:
                int_error += ten_norm(int_trbt - trbt, fro=fro)

            # angle_error = np.sum(np.abs(int_theta_c - int_theta))
            # zeta_error = np.sum(np.abs(int_zeta_c - int_zeta))
            # print (f'Iter {i}: (one-norm, error) = ({int_norm}, {int_error})  ----   (theta err, zeta err) = ({angle_error: 0.2e},{zeta_error: 0.2e})')
            print (f'Iter {i}: (one-norm, error) = ({int_norm}, {int_error})')
            

        x0, opt_state, loss = update_step(x0, opt_state)
        losses.append(float(loss))



        # print (loss)
        if verbose > 1 and i % 1000 == 0:
            print(f"Iteration {i}: Loss = {loss:.6e}")
        
        # Simple convergence check
        if i > 10 and abs(losses[-1] - losses[-2]) < 1e-12:
            if verbose > 1:
                print(f"Converged at iteration {i}")
            break
    final_params = pack_2dict(x0)
    L2_cost = _cost_vib(x0, obt, tbt, trbt, ob_sym_mats, ob_sym_vals, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none, trbt_is_None, 0, False)
    lam = _vib_vec_to_one_norm(x0, obt, tbt, trbt, ob_sym_mats, ob_sym_vals, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none, trbt_is_None)

    if verbose:
        print(f"\nInitial norm is {float(ten_norm(tbt, fro=fro)):.2e}")
        print(f"Finished THC factorization! Final norm of difference is {L2_cost:.2e}, 1-norm is {lam:.2f}\n")
        if obt_is_none:
            print(f"Note that one-norm does not include one-body component!")

        if include_bliss:
            print(f"BLISS included during optimization using {num_ob_syms} one-body symmetries")

    return final_params, lam






def get_vib_dfthc_v2(tbt, trbt=None, obt=None, ob_sym_list=[], Nthc=None, Nsig=None, regularize=True, maxiter=10000, initial_guess=None, learning_rate = 7.5e-3, verbose=True, fro=True):
    """
    Input:
        tbt: two body tensor of shape (Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal)
        trbt: three body tensor of shape (Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal)
        obt: one body tensor of shape (Nmode, Nmodal, Nmodal)
        Nthc: Number of THC orbitals
        Nsig: Number of singular values of the THC Zeta matrix
    """
    Nmode = tbt.shape[0]
    Nmodal = tbt.shape[-1]

    if Nthc is None:
        Nthc = Nmodal + 2
        if verbose:
            print(f"Using default THC rank of num_modals + 2 = {Nthc}")
    
    if Nsig is None:
        Nsig = int(np.ceil(Nthc/2))
        if verbose:
            print(f"Using default Zeta matrix rank of Nthc/2 = {Nsig}")

    #Bliss symmetries and their coefficients
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

    #params = _initialize_vib_params(initial_guess, tbt, Nthc, Nmodal, include_bliss, num_ob_syms)

    if initial_guess == None:
        params={}
        params['theta'] = np.random.uniform(low=0, high=2*np.pi, size=(Nmode, Nthc, Nmodal - 1))
        params['sigma'] = np.ones(Nsig)
        params['phi'] = np.random.uniform(low=0, high=2*np.pi, size=(Nsig, Nmode*Nthc-1))

        if trbt_is_None == False:
            params['gamma'] = np.zeros((Nmode, Nmode, Nmode, Nthc, Nthc, Nthc))
        print ("Initial guess is set to None")
    elif initial_guess == 'cp3':
        zeta, xi = _get_vib_cp3(tbt, Nthc, fro)
        zeta = np.transpose(zeta, (0,2,1,3))
        zeta_mat = zeta.reshape((Nmode*Nthc, Nmode*Nthc))
        sigmas, eig_vecs = np.linalg.eigh(zeta_mat)
        sigmas_srtd = sigmas[np.argsort(np.abs(sigmas))[:Nsig]]
        eig_vecs_srtd = eig_vecs[:, np.argsort(np.abs(sigmas))[:Nsig]]
        phi = normalized_xi_to_theta(eig_vecs_srtd.T)

        theta = normalized_xi_to_theta(xi)
        params={}
        params['theta'] = theta
        params['sigma'] = sigmas_srtd
        params['phi'] = phi
        if trbt_is_None == False:
            params['gamma'] = np.zeros((Nmode, Nmode, Nmode, Nthc, Nthc, Nthc))
        print ("Initial guess is set to cp3")
    elif isinstance(initial_guess, dict):
        params = copy(initial_guess)
        if trbt_is_None == False:
            if 'gamma' not in params:
                params['gamma'] = np.zeros((Nmode, Nmode, Nmode, Nthc, Nthc, Nthc))
        print ("Initial guess is user provided")
        


    if regularize is True:
        rho, params = _compute_vib_penalty_param(tbt, obt, ob_sym_list, Nthc, params, int(np.ceil(maxiter/10)))
        # if include_bliss:
        #     rho *= 2 / Nmode

        if verbose:
            print(f"Found regularization parameter rho={rho:.2e}")
    elif regularize is False or regularize is None:
        rho = 0
        if verbose:
            print(f"No regularization: setting rho=0")
    else:
        rho = regularize
        regularize=True
        if verbose:
            print(f"Regularization found: setting rho={rho:.2e}")

    def pack_2dict(x_vec):
        theta, sigma, phi, gamma, avec, bvec, beta_mats_params, dvec = _extract_from_x_vec(x_vec, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None)    
        my_dict = {"theta" : theta, "sigma" : sigma, "phi": phi}
        if trbt_is_None == False:
            my_dict["gamma"] = gamma
        if include_bliss:
            my_dict["avec"] = avec
            my_dict["bvec"] = bvec
            my_dict["beta_mats_params"] = beta_mats_params
            my_dict["dvec"] = dvec
        return my_dict

    def unpack_dict(my_dict):
        num_vars = Nmode * Nthc * (Nmodal-1) + Nsig + Nsig * (Nmode*Nthc-1)
        
        if trbt_is_None == False:
            num_vars += Nmode**3 * Nthc**3

        if include_bliss:
            num_vars += avec_len + bvec_len + beta_params_len + dvec_len

        x_vec = np.zeros(num_vars)

        theta_fin = Nmode*Nthc*(Nmodal-1)                                    
        x_vec[:theta_fin] = my_dict["theta"].flatten()

        sig_fin = theta_fin + Nsig
        x_vec[theta_fin:sig_fin] = my_dict["sigma"].flatten()

        phi_fin = sig_fin + Nsig*(Nmode*Nthc-1)
        x_vec[sig_fin:phi_fin] = my_dict["phi"].flatten()


        if trbt_is_None == False:
            G_fin = sig_fin + Nmode**3*Nthc**3
            x_vec[sig_fin:G_fin] = my_dict["gamma"].flatten()
            sig_fin = G_fin

        if include_bliss:
            a_fin = sig_fin + avec_len
            x_vec[sig_fin:a_fin] = my_dict["avec"]

            b_fin = a_fin + bvec_len
            x_vec[a_fin:b_fin] = my_dict["bvec"]

            beta_fin = b_fin + num_ob_syms*ob_mat_num_params
            x_vec[b_fin:beta_fin] = my_dict["beta_mats_params"].flatten()

            x_vec[beta_fin:] = my_dict["dvec"]

        return jnp.array(x_vec)

    @jit
    def cost_flat(x_vec):
        x00 = jnp.concatenate([x0_rest, x_vec])
        return _cost_vib(x00, obt, tbt, trbt, ob_sym_mats, ob_sym_vals, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none=obt_is_none, trbt_is_None=trbt_is_None, rho=rho, regularize=regularize)

    optimizer = optax.adam(learning_rate)
    x0_all = unpack_dict(params)
    theta_fin = Nmode*Nthc*(Nmodal-1)
    x0_rest = x0_all[:theta_fin]
    x0 = x0_all[theta_fin:]
    opt_state = optimizer.init(x0)

    @jit
    def update_step(x_vec, opt_state):
        loss, grads = jax.value_and_grad(cost_flat)(x_vec)
        updates, opt_state = optimizer.update(grads, opt_state, x_vec)
        x_vec = optax.apply_updates(x_vec, updates)
        return x_vec, opt_state, loss

    # Optimization loop
    losses = []
    org_diff = ten_norm(tbt, fro=fro)
    if trbt_is_None == False:
        org_diff += ten_norm(trbt, fro=fro)
    print ("Original tensor norm: ", org_diff)
    # int_theta, int_sigma, int_phi, int_gamma, _, _, _, _ = _extract_from_x_vec(x0_all, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None)
    for i in range(maxiter):
        #Check intermediate one-norm
        if  i % 100 == 0:
            # int_theta_c, int_sigma_c, int_phi_c = copy(int_theta), copy(int_sigma), copy(int_phi)
            int_norm = _vib_vec_to_one_norm(x0_all, obt, tbt, trbt, ob_sym_mats, ob_sym_vals, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none, trbt_is_None)
            int_theta, int_sigma, int_phi, int_gamma, _, _, _, _ = _extract_from_x_vec(x0_all, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, trbt_is_None)
            int_tbt, int_trbt = unfold_vib_thc(int_theta, int_sigma, int_phi, int_gamma, trbt_is_None)
            int_error = ten_norm(int_tbt - tbt, fro=fro)
            if trbt_is_None == False:
                int_error += ten_norm(int_trbt - trbt, fro=fro)

            # angle_error = np.sum(np.abs(int_theta_c - int_theta))
            # zeta_error = np.sum(np.abs(int_zeta_c - int_zeta))
            # print (f'Iter {i}: (one-norm, error) = ({int_norm}, {int_error})  ----   (theta err, zeta err) = ({angle_error: 0.2e},{zeta_error: 0.2e})')
            print (f'Iter {i}: (one-norm, error) = ({int_norm}, {int_error})')
            

        x0, opt_state, loss = update_step(x0, opt_state)
        x0_all = jnp.concatenate([x0_rest, x0])
        losses.append(float(loss))



        # print (loss)
        if verbose > 1 and i % 1000 == 0:
            print(f"Iteration {i}: Loss = {loss:.6e}")
        
        # Simple convergence check
        if i > 10 and abs(losses[-1] - losses[-2]) < 1e-12:
            if verbose > 1:
                print(f"Converged at iteration {i}")
            break
    final_params = pack_2dict(x0_all)
    L2_cost = _cost_vib(x0_all, obt, tbt, trbt, ob_sym_mats, ob_sym_vals, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none, trbt_is_None, 0, False)
    lam = _vib_vec_to_one_norm(x0_all, obt, tbt, trbt, ob_sym_mats, ob_sym_vals, Nthc, Nsig, Nmode, Nmodal, num_ob_syms, include_bliss, obt_is_none, trbt_is_None)

    if verbose:
        print(f"\nInitial norm is {float(ten_norm(tbt, fro=fro)):.2e}")
        print(f"Finished THC factorization! Final norm of difference is {L2_cost:.2e}, 1-norm is {lam:.2f}\n")
        if obt_is_none:
            print(f"Note that one-norm does not include one-body component!")

        if include_bliss:
            print(f"BLISS included during optimization using {num_ob_syms} one-body symmetries")

    return final_params, lam




