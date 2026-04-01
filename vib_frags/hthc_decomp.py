import numpy as np
import scipy as sp
import jax
from jax import numpy as jnp
from jax import grad, jit, vmap
from jax import lax
import optax
from functools import partial
import numpy as np
from copy import copy
from .tensor_utils import symmetrize_tbt, symmetrize_trbt, obt2_proj_mat
from .bliss_pauli import build_scatter_indices, pack_minimal_bliss_params, unpack_minimal_bliss_params_jax, get_bliss_hamiltonian_jax
from .bliss_utils import initialize_bliss_tensors

# Forcibly disable 64-bit execution
jax.config.update("jax_enable_x64", True)

# Optional: Force matmul operations to use the fastest possible float32 math
# jax.config.update("jax_default_matmul_precision", "tensorfloat32")


def _symmetrize_zeta(zeta):
    zeta_sym = (
        zeta + 
        np.transpose(zeta, (1,0,3,2))) /2
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



def zeta_dict_2_ten(zeta = dict):
    """
    Convert zeta given as a dictionary to zeta array. Note that if zeta_dict uses unsymmetrized mode-pairs, 
    the resulting zeta tensor will also be unsymmetrized

    zeta: dictionary
    """
    
    #Find Rthc
    Rthc = 0
    for (i,j), Zij in zeta.items():
        Rthc_i = max(Zij.shape)
        Rthc = Rthc_i if Rthc_i > Rthc else Rthc
    
    nmodes = np.max(np.array(list(zeta.keys()))) + 1
    zeta_ten = np.zeros((nmodes, nmodes, Rthc, Rthc))
    for (i,j), Zij in zeta.items():
        zeta_ten[i,j][:Zij.shape[0], :Zij.shape[1]] = Zij
    return zeta_ten


def theta_dict_2_ten(theta = dict):
    """
    Convert theta given as a dictionary to theta array.

    theta: dictionary
    nmodals: Number of modals
    """
    #Find Rthc
    Rthc = 0
    for i, th_i in theta.items():
        Nthc_i = th_i.shape[-1]
        Rthc = Nthc_i if Nthc_i > Rthc else Rthc
    
    nmodes = max(list(theta.keys())) + 1
    nmodals_1 = theta[0].shape[0]
    theta_ten = np.zeros((nmodes, nmodals_1, Rthc))
    for i, th_i in theta.items():
        theta_ten[i][:, :th_i.shape[-1]] = th_i
    return theta_ten



def unfold_vib_hthc(theta, zeta, gamma=None, trbt_is_None=True):
    Nmode = len(theta)
    Nmodal = theta[0].shape[0] + 1

    tbt_thc = np.zeros((Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal))
    xupq = {}
    for i, theta_i in theta.items():
        xi_i = angles_to_unit_vectors(theta_i)
        xupq[i] = np.einsum("pU,qU->Upq", xi_i, xi_i)

    for (i,j), Zij in zeta.items():
        tbt_thc[i,:,:,j,:,:] = np.einsum('UV,Upq,Vrs->pqrs', Zij, xupq[i], xupq[j])

    if not trbt_is_None:
        trbt_thc = np.zeros((Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal))
    else:
        trbt_thc = None
    return tbt_thc, trbt_thc





def theta_ten_to_unit_vectors(theta_ten):
    """
    Convert theta given as a (nmodes, nmodals-1, nthc) array to corresponding normalized vectors
    """
    nmodes, nmodals_1, Rthc = theta_ten.shape
    xi = np.zeros((nmodes, nmodals_1+1, Rthc))
    for i in range(nmodes):
        xi[i] = angles_to_unit_vectors(theta_ten[i])
    return xi







def THC_2_mat(xi, zeta):
    """
    Convert tbt defined by THC parameters to matrix representation in the one excitation per mode subspace.

    Parameters
    ----------
    xi : np.array
        THC obrital rotation matrix of shape (nmodes, nmodals, nthc)
    zeta : np.array
        THC zeta tensor (nmodes, nmodes, nthc, nthc) = (i,j,u,v)

    Returns
    -------
    sp.sparse.csc_matrix
        Two body tensor as a scipy sparse matrix of shape (nmodals^nmodes, nmodals^nmodes).
    """

    
    nmodes, nmodals, nthc = xi.shape
    id_mat = sp.sparse.identity(nmodals**nmodes, dtype=float, format='csc')

    tbt_mat = sp.sparse.csc_matrix((nmodals**nmodes, nmodals**nmodes), dtype=float)                     #Zero matrix

    obts = np.einsum('ipu, iqu -> iupq', xi, xi)
    for i in range(nmodes):
        for u in range (nthc):
            obtiu = np.zeros((nmodes, nmodals, nmodals), dtype=float)
            obtiu[i] = obts[i, u]
            obtiu_mat = obt2_proj_mat(obtiu)
            for j in range(nmodes):
                for v in range (nthc):
                    obtjv = np.zeros((nmodes, nmodals, nmodals), dtype=float)
                    obtjv[j] = obts[j, v]
                    obtjv_mat = obt2_proj_mat(obtjv)
                    if i != j:
                        tbt_mat += 0.25*zeta[i, j, u, v]*(id_mat - 2*obtiu_mat)*(id_mat - 2*obtjv_mat)

    return tbt_mat        



  



def full_THC_2_mat(H1, theta, zeta):
    """
    Obtain sparse matrix in the physical subspace of the full two mode coupling Hamiltonian after THC.
    This function can be used to test the THC factorization math by comparing the output with the unfactorized Hamiltonian matrix
    H1: one body tensor of the original Hamiltonian
    theta: THC angles stored as dictionary
    zeta: THC core tensor stored as dictionary with keys  (i,j) where i > j.
    """
    
    #tensor reconstruction
    H2_hthc_unsym,_ = unfold_vib_hthc(theta, zeta)
    H2_hthc = symmetrize_tbt(H2_hthc_unsym)

    #Hthc mat
    theta_ten = theta_dict_2_ten(theta)
    xi = theta_ten_to_unit_vectors(theta_ten)
    zeta_ten = zeta_dict_2_ten(zeta)

    H1_hthc = H1 + np.einsum('ipqjrr -> ipq', H2_hthc)
    H1_hthc_mat = obt2_proj_mat(H1_hthc)
    
    H2_hthc_mat = THC_2_mat(xi, zeta_ten)
    
    c_hthc = - (1/4)*np.einsum('ippjrr -> ', H2_hthc)
    c_hthc_mat = c_hthc*sp.sparse.identity(H1_hthc_mat.shape[0], format='csc')

    H_hthc_mat = c_hthc_mat + H1_hthc_mat + H2_hthc_mat

    return H_hthc_mat







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
    
    # Vectorized generation of xupq  
    # Apply function across all M blocks simultaneously
    xi = batched_angles_to_vectors(theta) 
    
    # Create xupq for all M blocks at once. 
    xupq = jnp.einsum("MpU,MqU->MUpq", xi, xi, optimize=True)

    # First Contraction: Contract U between zeta and xupq
    # zeta: (i, j, U, V), xupq: (i, U, p, q) -> intermediate: (i, j, V, p, q)
    intermediate = jnp.einsum('ijUV,iUpq->ijVpq', zeta, xupq, optimize=True)

    # Second Contraction: Contract V between intermediate and xupq
    # intermediate: (i, j, V, p, q), xupq: (j, V, r, s) 
    # We output to (i, p, q, j, r, s) to perfectly match the layout of tbt_full!
    recon = jnp.einsum('ijVpq,jVrs->ipqjrs', intermediate, xupq, optimize=True)

    # Global Loss Calculation
    # We subtract the entire reconstructed 6D tensor from the target at once.
    loss = jnp.sqrt(jnp.sum((tbt_full - recon)**2))
    
    return loss


@partial(jax.jit, static_argnums=(3)) 
def _tbt_error_opt(theta, zeta, tbt_full, norm_tbt_sq):
    # theta shape: (M, N-1, R)
    # zeta shape:  (M, M, R, R) -> axes: (i, j, U, V)
    
    # Vectorized generation of xupq
    xi = batched_angles_to_vectors(theta) 
    xupq = jnp.einsum("MpU,MqU->MUpq", xi, xi, optimize=True)

    # Flatten spatial dimensions for efficient BLAS execution
    M, R, p, q = xupq.shape
    P = p * q
    x_flat = xupq.reshape(M, R, P)
    
    # Flatten target tensor: (M, p, q, M, r, s) -> (M, P, M, P)
    tbt_flat = tbt_full.reshape(M, P, M, P)

    # Term 2: The Cross Term (-2 <T, R>)
    # Project T onto the x basis step-by-step to avoid large memory allocations
    Y = jnp.einsum('iPjQ,iUP->ijUQ', tbt_flat, x_flat, optimize=True)
    W = jnp.einsum('ijUQ,jVQ->ijUV', Y, x_flat, optimize=True)
    cross_term = jnp.einsum('ijUV,ijUV->', zeta, W, optimize=True)

    # Term 3: The Norm of the Reconstruction (||R||^2)
    # Calculate the spatial overlap (Gram) matrix first
    S = jnp.einsum('iUP,iAP->iUA', x_flat, x_flat, optimize=True)
    
    # Contract overlaps with zeta. 
    # Using Z_inter controls the contraction path strictly to smaller intermediate ranks
    Z_inter = jnp.einsum('ijUV,jVB->ijUB', zeta, S, optimize=True)
    R_norm_sq = jnp.einsum('ijUB,ijAB,iUA->', Z_inter, zeta, S, optimize=True)

    # Global Loss Calculation
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







def hthc_vib_one_norm(obt_full, tbt_full, trbt_full, zeta, gamma, obt_is_none, trbt_is_None, Rthc, bliss_params = None):
    #This function assumes the input zeta is a dictionary with keys as tuples of mode indices
    # optional argument bliss_params (a dictionary) allows for using precalculated one-body symmetry shift in finding the one-norm
    lambda_1 = 0.0
    sym_tbt_full = symmetrize_tbt(tbt_full)
    if trbt_is_None == False:
        sym_trbt_full = symmetrize_trbt(trbt_full)

    if obt_is_none == False:
        kappa = obt_full + (1)*np.einsum('ipqjrr -> ipq', sym_tbt_full)
        if trbt_is_None == False:
            kappa += (3/4)*np.einsum('ipqjrrktt -> ipq', sym_trbt_full)
        Nmode, Nmodal = kappa.shape[0], kappa.shape[-1]
        if (bliss_params is not None) and ('A' in bliss_params):
            _,kappa,_,_ = get_bliss_hamiltonian_jax(Nmode, Nmodal, kappa, None, None, bliss_params, 2)
        lambda_1 = (1/2)*np.sum(np.abs(np.linalg.eigvalsh(kappa)))

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





@partial(jax.jit, static_argnums=(5))
def _hthc_vib_one_norm(obt_full, tbt_full, trbt_full, zeta, gamma, trbt_is_None):
    # This function assumes the input zeta is an array obtained by padding zeros to the original zeta dictionary
    # Additionally, we assume tbt_full is symmetrized
    lambda_1 = 0.0
    kappa = obt_full + (1)*jnp.einsum('ipqjrr -> ipq', tbt_full)
    if trbt_is_None == False:
        kappa += (3/4)*jnp.einsum('ipqjrrktt -> ipq', trbt_full)
    lambda_1 = (1/2)*jnp.sum(jnp.abs(jnp.linalg.eigvalsh(kappa)))

    zeta_tilde = zeta
    if trbt_is_None == False:
        zeta_tilde += (3/2)*jnp.einsum('ijkUVW -> ijUV', gamma)

    # Factor of 1/4 below comes from converting number operator to reflection. Note, whether zeta is symmetrized or unsymmetrized, it does not change the one-norm expression here.
    lambda_2 = (1/4)*jnp.sum(jnp.abs(zeta_tilde))


    lambda_3 = 0
    if trbt_is_None == False:
        lambda_3 += (1/8)*jnp.sum(jnp.abs(gamma))

    return lambda_1, lambda_2, lambda_3




@partial(jax.jit, static_argnums=(2,3,4,5,6,7))
def _extract_from_x_vec(x_vec, indices, split_locs, Rthc, Nmode, Nmodal, bliss, trbt_is_None):
    theta_1d, zeta_1d, rest_1d = jnp.split(x_vec, [split_locs[0], split_locs[0] + split_locs[1]])

    # Initialize full arrays of zeros for theta and zeta
    theta = jnp.zeros((Nmode, Nmodal-1, Rthc))
    zeta = jnp.zeros((Nmode, Nmode, Rthc, Rthc))
    theta_indices, zeta_indices = indices['theta'], indices['zeta']
    
    # Scatter the 1D values into their specific structured locations
    theta = theta.at[theta_indices].set(theta_1d)
    zeta = zeta.at[zeta_indices].set(zeta_1d)

    # Do similar operations if trbt is not None
    if trbt_is_None == False:
        gamma_1d, rest_1d = jnp.split(rest_1d, [split_locs[2]])
        gamma = jnp.zeros((Nmode, Nmode, Nmode, Rthc, Rthc, Rthc))
        gamma_indices = indices['gamma']
        gamma = gamma.at[gamma_indices].set(gamma_1d)
        bl_idx = 3
    else:
        gamma = None   
        bl_idx = 2

    # Extract and unpack bliss parameters if present
    if bliss:
        re_scatter_mapppings = {k:(v, split_locs[bl_idx+i]) for i,(k,v) in enumerate(indices['bliss'].items())}
        bliss_params = unpack_minimal_bliss_params_jax(rest_1d, re_scatter_mapppings)
    else:
        bliss_params = None

    return theta, zeta, gamma, bliss_params 



@partial(jax.jit, static_argnums=(7, 8, 9, 10))
def _separate_costs(obt_full, tbt_full, trbt_full, theta, zeta, gamma, bliss_params, bliss, trbt_is_None, regularize, norm_tbt_sq):
    if bliss:
        mc = 2 if trbt_is_None else 3
        nmode, nmodal = tbt_full.shape[:2]
        c_new, obt_new, tbt_new, trbt_new = get_bliss_hamiltonian_jax(nmode, nmodal, obt_full, tbt_full, trbt_full, bliss_params, mc)
        fro_norm = _tbt_error(theta, zeta, tbt_new)*jnp.sqrt(2)                        #The sqrt(2) factor is used since we are using symmetrized tensors, but want the error for approximating unsymmetrized tensor.
        if regularize:
            one_norm = sum(_hthc_vib_one_norm(obt_new, tbt_new, trbt_new, zeta, gamma, trbt_is_None))
        else:
            one_norm = None
    else:
        fro_norm = _tbt_error_opt(theta, zeta, tbt_full, norm_tbt_sq)*jnp.sqrt(2)      #The sqrt(2) factor is used since we are using symmetrized tensors, but want the error for approximating unsymmetrized tensor.
        if regularize:
            one_norm = sum(_hthc_vib_one_norm(obt_full, tbt_full, trbt_full, zeta, gamma, trbt_is_None))
        else:
            one_norm = None

    # Trbt cost needs to be implemented
    return fro_norm, one_norm      





@partial(jax.jit, static_argnums=(5,6,7,8,9,10,11,12,13))
def _cost_vib(x_vec, obt_full, tbt_full, trbt_full, indices, split_locs, Rthc, Nmode, Nmodal, bliss, trbt_is_None, rho, regularize, norm_tbt_sq):
    theta, zeta, gamma, bliss_params = _extract_from_x_vec(x_vec, indices, split_locs, Rthc, Nmode, Nmodal, bliss, trbt_is_None)
    
    tot_cost, one_norm = _separate_costs(obt_full, tbt_full, trbt_full, theta, zeta, gamma, bliss_params, bliss, trbt_is_None, regularize, norm_tbt_sq)

    if regularize:
        tot_cost += rho * one_norm

    # jax.debug.print('fro_norm: {fro_norm}, one_norm: {one_norm}', fro_norm=fro_norm, one_norm=one_norm)
    return tot_cost





def initialize_thc_params(Nmode, Nmodal, Nthc, param_keys, initial_guess=None, trbt_is_None=True, bliss=False, random=False):
    """
    Initialize THC (or more generally BLISS THC) parameters.
    """
    if initial_guess is None:
        theta={}
        for i in range(Nmode):
            theta_r = np.random.uniform(low=-np.pi, high=np.pi, size=(Nmodal - 1, Nthc[i]))
            theta[i] = theta_r
        zeta={}
        for i in range(Nmode):
            for j in range(i):
                zeta[(i,j)] = np.zeros((Nthc[i], Nthc[j]))
        params={'theta': theta, 'zeta': zeta}
        if trbt_is_None == False:
            gamma = {}
            for i in range(Nmode):
                for j in range(i):
                    for k in range (j):
                        gamma[(i,j,k)] = np.zeros((Nthc[i], Nthc[j], Nthc[k]))
        if bliss:
            bliss_params = initialize_bliss_tensors(Nmode, Nmodal, param_keys, random)
            params.update(bliss_params)
        else:
            bliss_params = {}
    elif isinstance(initial_guess, dict):
        params = copy(initial_guess)
        if trbt_is_None == False:
            if 'gamma' not in params:
                params['gamma'] = np.zeros((Nmode, Nmode, Nmode, Nthc, Nthc, Nthc))
        if bliss:
            bliss_params = {}
            for sym in param_keys:
                if sym in params:
                    bliss_params[sym] = params[sym]
                else:
                    new_bliss_param = initialize_bliss_tensors(Nmode, Nmodal, [sym], random)
                    params.update(new_bliss_param)
                    bliss_params.update(new_bliss_param)
        else:
            bliss_params = {}
    else:
        raise ValueError("initial_guess needs to be either None or a dictionary.")
    return params, bliss_params








def get_vib_hthc(obt, tbt, trbt=None, Nthc=None, regularize=True, bliss=False, maxiter=10000, initial_guess=None, syms = None | list, random=False, learning_rate = 7.5e-3, verbose=True, chunk_size = 200):
    """
    Function to perform heterogeneous THC i.e. number of THC orbitals can be different for different modes.
    Input:
    ------
    tbt: Two body tensor of shape (Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal). This is assumed to be unsymmetrized. So pass H2_nonsym here.
    trbt: Three body tensor of shape (Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal, Nmode, Nmodal, Nmodal)
    obt: One body tensor of shape (Nmode, Nmodal, Nmodal)
    Nthc: list of number of THC orbitals for each mode
    bliss: If True, the code performs BLISS THC
    initial_guess: Initial guess for THC provided as a dictionary
    random: If True, when bliss is True and initial_guess is None, BLISS parameters are initialized to random values
    syms: List of symmetries to be included in the Killer operator. It could be one of the non-empty subsets of ['A', 'B', 'C', 'D', 'E', 'F'].
    """

    Nmode = tbt.shape[0]
    Nmodal = tbt.shape[-1]

    if Nthc is None:
        Nthc = (int(np.ceil(Nmodal+1)),)*Nmode
        if verbose:
            print(f"Using default homogeneous THC rank of ceil(num_modals+1) = {Nthc}")
    elif type(Nthc) is int:
        Nthc = (Nthc,)*Nmode
        print (f"Performing homogeneous THC")
    elif type(Nthc) is list:
        if len(Nthc) != Nmode:
            raise ("Nthc must be of length Nmode if not an integer")
        Nthc = tuple(Nthc)
    Rthc = max(Nthc)


    #___________________________________________________________________________________________________________________________________
    #___________________________________________Bliss symmetries and their coefficients_________________________________________________
    mc = 3 if trbt is not None else 2 
    param_keys = None
    if bliss:
        # Pre-compute bliss static index maps for dynamic JAX reconstruction
        if isinstance(syms, list):
            # Ensuring unwanted symmetries are removed
            if mc < 3:
                if 'D' in syms: syms.remove('D')
                if 'E' in syms: syms.remove('E')
                if 'F' in syms: syms.remove('F')
            if tbt is None:
                if 'B' in syms: syms.remove('B')
                if 'C' in syms: syms.remove('C')
            if obt is None:
                if 'A' in syms: syms.remove('A')
            param_keys = syms
        else:
            if mc == 2: 
                param_keys = ['A', 'B', 'C']
            elif mc == 3:
                param_keys = ['A', 'B', 'C', 'D', 'E', 'F']
        scatter_mappings = build_scatter_indices(Nmode, Nmodal, param_keys)

    if obt is None:
        raise ("One body tensor must be provided")

    if trbt is None:
        trbt_is_None = True
        trbt_full = None
    else:
        trbt_is_None = False

    #___________________________________________________________________________________________________________________________________
    #___________________________________________________________________________________________________________________________________




    #___________________________________________________________________________________________________________________________________
    #_________________________________________________Setting up initial guess and rho__________________________________________________
    params, bliss_params = initialize_thc_params(Nmode, Nmodal, Nthc, param_keys, initial_guess, trbt_is_None, bliss, random=random)
    if regularize is False or regularize is None:
        rho = 0
        if verbose:
            print(f"No regularization: setting rho=0")
    elif type(regularize) is float or type(regularize) is int:
        rho = regularize
        regularize=True
        if verbose:
            print(f"Using regularization constant, rho={rho:.2e}")
    elif regularize is True:
        rho = 1e-3
        if verbose:
            print(f"Using default regularization constant, rho={rho:.2e}")

    #___________________________________________________________________________________________________________________________________
    #___________________________________________________________________________________________________________________________________


    def build_zeta_index_map(zeta_dict):
        """
        Creates a flat 1D array of zeta values and a tuple of indices
        mapping them to a padded array. This allows reconstructing the full symmetrized zeta tensor using unsymmetrized zeta_dict.
        So zeta_dict is assumed to be the zeta tensor for an unsymmetrized tbt such as H2_nonsym
        """
        flat_values = []
        i_idx, j_idx, u_idx, v_idx = [], [], [], []
        
        for (i, j), Zij in zeta_dict.items():
            U, V = Zij.shape
            for u in range(U):
                for v in range(V):
                    flat_values.extend([Zij[u, v]/2, Zij[u, v]/2])
                    i_idx.extend([i, j])
                    j_idx.extend([j, i])
                    u_idx.extend([u, v])
                    v_idx.extend([v, u])
                    
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
    theta_size, zeta_size = theta_vec.shape[0], zeta_vec.shape[0]
    split_locs = (theta_size, zeta_size)
    indices = {'theta': theta_indices, 'zeta': zeta_indices}

    if bliss:
       bliss_params_1d = jnp.array(pack_minimal_bliss_params(bliss_params, scatter_mappings))
       x0 = jnp.concatenate([x0, bliss_params_1d])
       split_locs += tuple([scatter_mappings[k][1] for k in scatter_mappings.keys()])
       scatter_mappings = {k: (jnp.array(v[0]), v[1]) for k, v in scatter_mappings.items()}
       indices['bliss'] = {k:v[0] for k,v in scatter_mappings.items()}


    def pack_2dict(theta, zeta, Nthc, Nmode, gamma=None, bliss_params=None):
        theta_c = {i: np.array(theta[i, :, :Nthc[i]]) for i in range(Nmode)}
        zeta_c = {(i,j): np.array(zeta[i, j, :Nthc[i], :Nthc[j]])*2 for i in range (Nmode) for j in range(i)}       # The factor of 2 is to map symmetrized zeta to unsymmetrized zeta
        my_dict = {"theta" : theta_c, "zeta" : zeta_c, "Nthc": Nthc}
        if trbt_is_None == False:
            my_dict["gamma"] = gamma
        if bliss_params is not None:
            my_dict.update(bliss_params)
        return my_dict

    # We will pass symmetrized tbt to the optimizer
    jnp_tbt = jnp.array(symmetrize_tbt(tbt))
    jnp_obt = jnp.array(obt)

    # Precompute the constant target norm ---
    norm_tbt_sq = jnp.sum(jnp_tbt**2).item()

    @jit
    def cost_flat(x_vec):
        return _cost_vib(x_vec, jnp_obt, jnp_tbt, trbt_full, indices, split_locs, Rthc, Nmode, Nmodal, bliss, trbt_is_None, rho, regularize, norm_tbt_sq)
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(x0)

    # Compile chunk_size iterations into a single XLA execution ---
    
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
    org_diff = ten_norm(tbt, fro=True)
    if trbt_is_None == False:
        org_diff += ten_norm(trbt, fro=True)
    print ("Original tensor norm: ", org_diff, flush = True)

   # Calculate how many chunk_size-step blocks to run
    num_chunks = maxiter // chunk_size

    for chunk in range(num_chunks):
        # i maps to 0, 100, 200, 300, etc. if chunk_size = 100
        i = chunk * chunk_size
        
        # Evaluate and print intermediate state BEFORE the chunk runs
        int_theta, int_zeta, int_gamma, int_bliss_params = _extract_from_x_vec(x0, indices, split_locs, Rthc, Nmode, Nmodal, bliss, trbt_is_None)
        fro_norm, one_norm = _separate_costs(jnp_obt, jnp_tbt, trbt, int_theta, int_zeta, int_gamma, int_bliss_params, bliss, trbt_is_None, regularize, norm_tbt_sq)
        one_norm = 0.0 if one_norm is None else one_norm.item()
        print (f'Iter {i}: (error, 1-norm) = ({fro_norm.item()}, {one_norm})', flush = True)

        # Execute chunk_size steps
        x0, opt_state, chunk_losses = update_chunk(x0, opt_state)
        
        # Pull the chunk_size losses back to Python memory once
        losses.extend(np.array(chunk_losses).tolist())
        
        # # Simple convergence check (using the last two elements of the entire history)
        # if len(losses) > 1 and abs(losses[-1] - losses[-2]) < 1e-12:
        #     if verbose == True:
        #         print(f"Converged around iteration {i + chunk_size}", flush = True)
        #     break
        
    theta, zeta, gamma, bliss_params = _extract_from_x_vec(x0, indices, split_locs, Rthc, Nmode, Nmodal, bliss, trbt_is_None)   
    final_params = pack_2dict(theta, zeta, Nthc, Nmode, gamma, bliss_params)
    fro_norm, one_norm = _separate_costs(jnp_obt, jnp_tbt, trbt, theta, zeta, gamma, bliss_params, bliss, trbt_is_None, regularize, norm_tbt_sq)
    one_norm = 0.0 if one_norm is None else one_norm.item()
    if verbose:
        print(f"\nInitial norm is {float(ten_norm(tbt, fro=True)):.2e}")
        print(f"Finished THC factorization! Final norm of difference is {fro_norm.item():.2e}, 1-norm is {one_norm:.2e}\n")

    return final_params, one_norm