import numpy as np
import itertools
from opt_einsum import contract

# =============================================================================
# Utilities
# =============================================================================

def apply_zero_diagonal_mask_2d(tensor):
    """Ensures tensor is zero when i == j."""
    nmodes = tensor.shape[0]
    for i in range(nmodes):
        tensor[i, i, ...] = 0.0
    return tensor

def apply_zero_diagonal_mask_3d(tensor):
    """Ensures tensor is zero when i==j, j==k, or i==k."""
    nmodes = tensor.shape[0]
    for i in range(nmodes):
        for j in range(nmodes):
            for k in range(nmodes):
                if i == j or j == k or i == k:
                    tensor[i, j, k, ...] = 0.0
    return tensor

def initialize_bliss_tensors(nmodes, nmodals, param_keys, random=True):
    """
    Initializes a dictionary of Killer coefficient tensors ('A' through 'F')
    with random values strictly enforcing the necessary symmetries and zero-diagonals.
    """
    params = {}
    
    def _initializer(shape):
      if random:
        return np.random.default_rng().standard_normal(shape)
      else:
        return np.zeros(shape)

    if 'A' in param_keys:
        params['A'] = _initializer((nmodes,))
        
    if 'B' in param_keys:
        B = _initializer((nmodes, nmodes))
        B = 0.5 * (B + B.T)
        params['B'] = apply_zero_diagonal_mask_2d(B)

    if 'C' in param_keys:    
        C = _initializer((nmodes, nmodes, nmodals, nmodals))
        C = 0.5 * (C + np.einsum('ijrs->ijsr', C))
        params['C'] = apply_zero_diagonal_mask_2d(C)
        
    if 'D' in param_keys:
        D = _initializer((nmodes, nmodes, nmodes))
        D_sym = np.zeros_like(D)
        for perm in itertools.permutations([0, 1, 2]):
            D_sym += np.transpose(D, perm)
        D_sym /= 6.0
        params['D'] = apply_zero_diagonal_mask_3d(D_sym)

    if 'E' in param_keys:        
        E = _initializer((nmodes, nmodes, nmodes, nmodals, nmodals))
        E = 0.5 * (E + np.einsum('jikt...->ijkt...', E))
        E = 0.5 * (E + np.einsum('...ut->...tu', E))
        params['E'] = apply_zero_diagonal_mask_3d(E)
    
    if 'F' in param_keys:
        F = _initializer((nmodes, nmodes, nmodes, nmodals, nmodals, nmodals, nmodals))
        F = 0.5 * (F + np.einsum('ikjturs->ijkrstu', F))
        F = 0.5 * (F + np.einsum('ijksrut->ijkrstu', F))
        params['F'] = apply_zero_diagonal_mask_3d(F)
        
    return params

def pack_parameters(param_dict):
    """Flattens a parameter dictionary into a 1D array."""
    flat_list = []
    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        if key in param_dict:
            flat_list.append(param_dict[key].flatten())
    return np.concatenate(flat_list) if flat_list else np.array([])

def unpack_parameters(params_1d, nmodes, nmodals, mc):
    """
    Reconstructs the highly symmetric, zero-diagonal coefficient dictionary 
    from a 1D array, enforcing exact structural and Hermiticity symmetries.
    """
    param_dict = {}
    offset = 0
    
    if mc >= 1:
        size_A = nmodes
        param_dict['A'] = params_1d[offset : offset + size_A].reshape((nmodes,))
        offset += size_A
        
    if mc >= 2:
        size_B = nmodes**2
        B_raw = params_1d[offset : offset + size_B].reshape((nmodes, nmodes))
        B_sym = 0.5 * (B_raw + B_raw.T)
        param_dict['B'] = apply_zero_diagonal_mask_2d(B_sym)
        offset += size_B
        
        size_C = nmodes**2 * nmodals**2
        C_raw = params_1d[offset : offset + size_C].reshape((nmodes, nmodes, nmodals, nmodals))
        C_sym = 0.5 * (C_raw + np.einsum('ijrs->ijsr', C_raw))
        param_dict['C'] = apply_zero_diagonal_mask_2d(C_sym)
        offset += size_C
        
    if mc >= 3:
        size_D = nmodes**3
        D_raw = params_1d[offset : offset + size_D].reshape((nmodes, nmodes, nmodes))
        D_sym = np.zeros_like(D_raw)
        for perm in itertools.permutations([0, 1, 2]):
            D_sym += np.transpose(D_raw, perm)
        D_sym /= 6.0
        param_dict['D'] = apply_zero_diagonal_mask_3d(D_sym)
        offset += size_D
        
        size_E = nmodes**3 * nmodals**2
        E_raw = params_1d[offset : offset + size_E].reshape((nmodes, nmodes, nmodes, nmodals, nmodals))
        E_sym = 0.5 * (E_raw + np.einsum('jikt...->ijkt...', E_raw))
        E_sym = 0.5 * (E_sym + np.einsum('...ut->...tu', E_sym))
        param_dict['E'] = apply_zero_diagonal_mask_3d(E_sym)
        offset += size_E
        
        size_F = nmodes**3 * nmodals**4
        F_raw = params_1d[offset : offset + size_F].reshape((nmodes, nmodes, nmodes, nmodals, nmodals, nmodals, nmodals))
        F_sym = 0.5 * (F_raw + np.einsum('ikjturs->ijkrstu', F_raw))
        F_sym = 0.5 * (F_sym + np.einsum('ijksrut->ijkrstu', F_sym))
        param_dict['F'] = apply_zero_diagonal_mask_3d(F_sym)
        offset += size_F
        
    return param_dict




# =============================================================================
# Core Physics Logic
# =============================================================================

def construct_killer(nmodes, nmodals, mc, params=None):
    """
    Computes the constant, one-body, two-body, and three-body corrections
    generated by the Killer operator.
    
    The output tensors are strictly interleaved to match the physical Hamiltonian
    index ordering: mode, orbital, orbital, mode, orbital, orbital, etc.
    (e.g., 'ipqjrs' for 2-body and 'ipqjrsktu' for 3-body).
    
    Parameters:
    -----------
    nmodes : int
        Number of vibrational modes (sites).
    nmodals : int
        Number of basis functions (orbitals) per mode.
    mc : int
        Mode coupling level (1, 2, or 3). Determines the maximum rank of 
        the Killer operator terms to construct.
    params : dict, np.ndarray, or None, optional
        The coefficient parameters. If None, symmetrically valid random tensors 
        are generated automatically.
        If np.ndarray, it is assumed to have parameters corresponding to all symmetries
        at a given mc lvel.
        
    Returns:
    --------
    dc : float
        The constant correction (\Delta c).
    dh : np.ndarray
        The 1-body correction (\Delta h) of shape (nmodes, nmodals, nmodals).
    dg : np.ndarray or None
        The 2-body correction (\Delta g) of shape (nmodes, nmodals, nmodals, 
        nmodes, nmodals, nmodals). None if mc < 2.
    dv : np.ndarray or None
        The 3-body correction (\Delta v) of shape (nmodes, nmodals, nmodals, 
        nmodes, nmodals, nmodals, nmodes, nmodals, nmodals). None if mc < 3.
    """
    
    # 1. Parameter resolution
    if params is None:
        param_dict = initialize_bliss_tensors(nmodes, nmodals, mc)
    elif isinstance(params, np.ndarray):
        param_dict = unpack_parameters(params, nmodes, nmodals, mc)
    elif isinstance(params, dict):
        param_dict = params
    else:
        raise ValueError("params must be None, a 1D np.ndarray, or a dictionary.")
        
    eye = np.eye(nmodals)
    
    # Initialize corrections
    dc = 0.0
    dh = np.zeros((nmodes, nmodals, nmodals))
    dg = None
    dv = None
    
    if mc >= 2:
        dg = np.zeros((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
    if mc >= 3:
        dv = np.zeros((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
        
    # -------------------------------------------------------------------------
    # 1-Body Space Couplings (mc >= 1)
    # -------------------------------------------------------------------------
    if mc >= 1 and 'A' in param_dict:
        A = param_dict['A']
        dc -= np.sum(A)
        dh += np.einsum('i,pq->ipq', A, eye)

    # -------------------------------------------------------------------------
    # 2-Body Space Couplings (mc >= 2)
    # Target shape indices: 'ipqjrs'
    # -------------------------------------------------------------------------
    if mc >= 2:
        if 'B' in param_dict:
            B = param_dict['B']
            dc -= np.sum(B)
            dg += np.einsum('ij,pq,rs->ipqjrs', B, eye, eye)
        if 'C' in param_dict:
            C = param_dict['C']
            dh -= np.einsum('jipq->ipq', C)
            dg += 0.5 * (np.einsum('ijrs,pq->ipqjrs', C, eye) + np.einsum('jipq,rs->ipqjrs', C, eye))


    # -------------------------------------------------------------------------
    # 3-Body Space Couplings (mc >= 3)
    # Target shape indices: 'ipqjrsktu'
    # -------------------------------------------------------------------------
    if mc >= 3:
        if 'D' in param_dict:
            D = param_dict['D']
            dc -= np.sum(D)
            dv += np.einsum('ijk,pq,rs,tu->ipqjrsktu', D, eye, eye, eye)
        if 'E' in param_dict:
            E = param_dict['E']
            dh -= np.einsum('jkipq->ipq', E)
            dv += (np.einsum('ijktu,pq,rs->ipqjrsktu', E, eye, eye) + 
               np.einsum('ikjrs,pq,tu->ipqjrsktu', E, eye, eye) + 
               np.einsum('kjipq,rs,tu->ipqjrsktu', E, eye, eye)) / 3.0
        if 'F' in param_dict:
            F = param_dict['F']
            dg -= np.einsum('kijpqrs->ipqjrs', F)
            dv += (np.einsum('ijkrstu,pq->ipqjrsktu', F, eye) + 
               np.einsum('jikpqtu,rs->ipqjrsktu', F, eye) + 
               np.einsum('kjirspq,tu->ipqjrsktu', F, eye)) / 3.0   

    return dc, dh, dg, dv




def get_bliss_hamiltonian(obt, tbt=None, trbt=None, params=None):
    """
    Applies the Killer operator transformation to the physical Hamiltonian.
    H_{new} = H_{old} - K.
    
    Parameters:
    -----------
    obt : np.ndarray
        The 1-body Hamiltonian tensor (h), shape (nmodes, nmodals, nmodals).
    tbt : np.ndarray, optional
        The 2-body Hamiltonian tensor (g), shape interleaved as 
        (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals).
    trbt : np.ndarray, optional
        The 3-body Hamiltonian tensor (v), shape interleaved as
        (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals).
    params : dict, np.ndarray, or None, optional
        The coefficient parameters.
        
    Returns:
    --------
    c_new : float
        The constant shift applied to the energy.
    obt_new : np.ndarray
        The transformed 1-body tensor (h^B).
    tbt_new : np.ndarray or None
        The transformed 2-body tensor (g^B).
    trbt_new : np.ndarray or None
        The transformed 3-body tensor (v^B).
    """
    nmodes, nmodals, _ = obt.shape
    
    # Infer Mode Coupling (mc) level based on provided tensors
    if trbt is not None:
        mc = 3
    elif tbt is not None:
        mc = 2
    else:
        mc = 1
        
    # Generate the \Delta K corrections (initializes params if None)
    dc, dh, dg, dv = construct_killer(nmodes, nmodals, mc, params)
    
    # Apply transformation: H_{new} = H_{old} - K_correction
    c_new = -dc
    obt_new = obt - dh
    
    tbt_new = None
    if mc >= 2 and tbt is not None:
        tbt_new = tbt - dg
        
    trbt_new = None
    if mc >= 3 and trbt is not None:
        trbt_new = trbt - dv
        
    return c_new, obt_new, tbt_new, trbt_new