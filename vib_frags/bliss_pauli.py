import numpy as np
import jax
import jax.numpy as jnp
import optax
import itertools

# =============================================================================
# Advanced Indexing & Minimal Parameter Utilities
# =============================================================================

def build_scatter_indices(nmodes, nmodals, param_keys):
    """
    Computes the mapping between dense symmetric tensors and a strictly minimal 
    1D array. By assigning unique IDs to elements and projecting them through 
    the symmetry operations, we construct a scatter index map.
    
    This avoids optimizer "dead weight" while relying strictly on fast JAX array 
    indexing during the forward pass.
    """
    scatter_mappings = {}
    
    def get_minimal_mapping(sym_tensor):
      """Helper to extract unique elements and build the 1D mapping index array."""
      flat_sym = sym_tensor.flatten()
      valid_unique = np.unique(flat_sym[flat_sym != -1])
      num_params = len(valid_unique)
      
      # Mapping: -1 (zeros) maps to index 0. Valid parameters map to 1...num_params.
      val_to_idx = {val: i + 1 for i, val in enumerate(valid_unique)}
      val_to_idx[-1] = 0
      
      scatter_flat = np.array([val_to_idx[val] for val in flat_sym])
      return scatter_flat.reshape(sym_tensor.shape), num_params

    # A: General 1D array (No zero masking needed)
    if 'A' in param_keys:
        A_ids = np.arange(nmodes)
        scatter_mappings['A'] = get_minimal_mapping(A_ids)
        
    # B & C: 2-Body Couplings (Mask diagonal i == j)
    if 'B' in param_keys:
        B_ids = np.arange(nmodes**2).reshape(nmodes, nmodes)
        B_sym = np.minimum(B_ids, B_ids.T)
        for i in range(nmodes): B_sym[i, i] = -1
        scatter_mappings['B'] = get_minimal_mapping(B_sym)
        
    if 'C' in param_keys:
        C_ids = np.arange(nmodes**2 * nmodals**2).reshape(nmodes, nmodes, nmodals, nmodals)
        C_sym = np.minimum(C_ids, np.einsum('ijrs->ijsr', C_ids))
        for i in range(nmodes): C_sym[i, i, :, :] = -1
        scatter_mappings['C'] = get_minimal_mapping(C_sym)
        
    # D, E, F: 3-Body Couplings (Mask diagonals i==j, j==k, i==k)
    if 'D' in param_keys or 'E' in param_keys or 'F' in param_keys:
        D_ids = np.arange(nmodes**3).reshape(nmodes, nmodes, nmodes)
        D_sym = D_ids.copy()
        for perm in itertools.permutations([0, 1, 2]):
            D_sym = np.minimum(D_sym, np.transpose(D_ids, perm))
            
        E_ids = np.arange(nmodes**3 * nmodals**2).reshape(nmodes, nmodes, nmodes, nmodals, nmodals)
        E_sym = np.minimum(E_ids, np.transpose(E_ids, (1, 0, 2, 3, 4))) # Site symmetry i,j
        E_sym = np.minimum(E_sym, np.transpose(E_sym, (0, 1, 2, 4, 3))) # Hermiticity t,u
        
        F_ids = np.arange(nmodes**3 * nmodals**4).reshape(nmodes, nmodes, nmodes, nmodals, nmodals, nmodals, nmodals)
        F_C  = np.transpose(F_ids, (0, 2, 1, 5, 6, 3, 4)) # Composite swap
        F_H  = np.transpose(F_ids, (0, 1, 2, 4, 3, 6, 5)) # Simultaneous orbital swap
        F_CH = np.transpose(F_H,   (0, 2, 1, 5, 6, 3, 4)) # Commuted operation
        F_sym = np.minimum(np.minimum(F_ids, F_C), np.minimum(F_H, F_CH))
        
        # Apply 3-body hollow mask
        for i in range(nmodes):
            for j in range(nmodes):
                for k in range(nmodes):
                    if i == j or j == k or i == k:
                        D_sym[i, j, k] = -1
                        E_sym[i, j, k, ...] = -1
                        F_sym[i, j, k, ...] = -1

        if 'D' in param_keys:           
            scatter_mappings['D'] = get_minimal_mapping(D_sym)
        if 'E' in param_keys:
            scatter_mappings['E'] = get_minimal_mapping(E_sym)
        if 'F' in param_keys:
            scatter_mappings['F'] = get_minimal_mapping(F_sym)
        
    return scatter_mappings

def pack_minimal_bliss_params(dense_dict, scatter_mappings):
    """
    Packs a dense dictionary of symmetric tensors into the minimal 1D parameter array 
    expected by the optimizer, avoiding all redundant entries.
    """
    flat_list = []
    for key in scatter_mappings.keys():           
        scatter_idx, num_params = scatter_mappings[key]
        p_slice = np.zeros(num_params)
        flat_dense = dense_dict[key].flatten()
        flat_scatter = scatter_idx.flatten()
        
        # Extract the first occurrence of each unique parameter
        for k in range(1, num_params + 1):
            idx = np.where(flat_scatter == k)[0][0]
            p_slice[k-1] = flat_dense[idx]
            
        flat_list.append(p_slice)
        
    return np.concatenate(flat_list) if flat_list else np.array([])

def unpack_minimal_bliss_params_jax(params_1d, scatter_mappings):
    """
    Rebuilds the mathematically exact, fully symmetric dense tensors during the JAX 
    forward pass. Uses lightning-fast advanced integer indexing.
    """
    param_dict = {}
    offset = 0
    
    for key in scatter_mappings.keys():    
        scatter_idx, num_params = scatter_mappings[key]
        
        # Extract the segment belonging to this tensor
        p_slice = params_1d[offset : offset + num_params]
        
        # Prepend a 0.0 to act as the target for all masked/zero elements (index 0)
        p_padded = jnp.concatenate([jnp.array([0.0]), p_slice])
        
        # Broadcast via advanced indexing into the full dense shape
        param_dict[key] = p_padded[scatter_idx]
        offset += num_params
        
    return param_dict

# =============================================================================
# JAX-Compatible Core Physics & Loss Functions
# =============================================================================

def construct_killer_jax(nmodes, nmodals, mc, param_dict):
    """JAX implementation of the Killer operator terms."""
    eye = jnp.eye(nmodals)
    
    dc = 0.0
    dh = jnp.zeros((nmodes, nmodals, nmodals))
    dg = None if mc < 2 else jnp.zeros((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
    dv = None if mc < 3 else jnp.zeros((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
    
    if mc >= 1 and 'A' in param_dict:
        A = param_dict['A']
        dc -= jnp.sum(A)
        dh += jnp.einsum('i,pq->ipq', A, eye)

    if mc >= 2:
        if 'B' in param_dict:
            B = param_dict['B']
            dc -= jnp.sum(B)
            dg += jnp.einsum('ij,pq,rs->ipqjrs', B, eye, eye)
        if 'C' in param_dict:
            C = param_dict['C']
            dh -= jnp.einsum('jipq->ipq', C)
            dg += 0.5 * (jnp.einsum('ijrs,pq->ipqjrs', C, eye) + jnp.einsum('jipq,rs->ipqjrs', C, eye))

    if mc >= 3:
        if 'D' in param_dict:
            D = param_dict['D']
            dc -= jnp.sum(D)
            dv += jnp.einsum('ijk,pq,rs,tu->ipqjrsktu', D, eye, eye, eye)
        if 'E' in param_dict:
            E = param_dict['E']
            dh -= jnp.einsum('jkipq->ipq', E)
            dv += (jnp.einsum('ijktu,pq,rs->ipqjrsktu', E, eye, eye) + 
               jnp.einsum('ikjrs,pq,tu->ipqjrsktu', E, eye, eye) + 
               jnp.einsum('kjipq,rs,tu->ipqjrsktu', E, eye, eye)) / 3.0
        if 'F' in param_dict:
            F = param_dict['F']
            dg -= jnp.einsum('kijpqrs->ipqjrs', F)
            dv += (jnp.einsum('ijkrstu,pq->ipqjrsktu', F, eye) + 
               jnp.einsum('jikpqtu,rs->ipqjrsktu', F, eye) + 
               jnp.einsum('kjirspq,tu->ipqjrsktu', F, eye)) / 3.0    
    return dc, dh, dg, dv

def get_bliss_hamiltonian_jax(nmodes, nmodals, obt, tbt, trbt, param_dict, mc):
    """JAX native version of the Hamiltonian subtraction transformation i.e. H-K."""
    dc, dh, dg, dv = construct_killer_jax(nmodes, nmodals, mc, param_dict)
    
    c_new = -dc
    obt_new = obt - dh  if (mc >= 1 and obt is not None) else None
    tbt_new = (tbt - dg) if (mc >= 2 and tbt is not None) else None
    trbt_new = (trbt - dv) if (mc >= 3 and trbt is not None) else None
    
    return c_new, obt_new, tbt_new, trbt_new




def get_bliss_hamiltonian_inplace_jax(obt, tbt=None, trbt=None, param_dict=None, mc=2):
    """
    Directly injects the Killer parameters into the Hamiltonian tensors, 
    avoiding the construction of massive sparse Kronecker-delta tensors.
    """
    if param_dict is None:
        param_dict = {}

    nmodes, nmodals, _ = obt.shape
    c_new = 0.0
    obt_new = obt
    
    # Group mode indices at the front so vmap sequential stripping works cleanly.
    # We will un-group them back to interleaved format at the very end.
    tbt_grouped = None
    trbt_grouped = None
    
    if tbt is not None:
        # i, p, q, j, r, s  ->  i, j, p, q, r, s
        tbt_grouped = tbt.transpose(0, 3, 1, 2, 4, 5)
        
    if trbt is not None:
        # i, p, q, j, r, s, k, t, u  ->  i, j, k, p, q, r, s, t, u
        trbt_grouped = trbt.transpose(0, 3, 6, 1, 2, 4, 5, 7, 8)

    # =========================================================================
    # Constant Shift Space
    # =========================================================================
    if mc >= 1 and 'A' in param_dict:
        c_new += jnp.sum(param_dict['A'])
    if mc >= 2 and 'B' in param_dict:
        c_new += jnp.sum(param_dict['B'])
    if mc >= 3 and 'D' in param_dict:
        c_new += jnp.sum(param_dict['D'])

    # =========================================================================
    # 1-Body Space
    # =========================================================================
    if mc >= 1 and 'A' in param_dict:
        def inject_A(h_i, A_i):
            p = jnp.arange(nmodals)
            return h_i.at[p, p].add(-A_i)
        obt_new = jax.vmap(inject_A)(obt_new, param_dict['A'])

    if mc >= 2 and 'C' in param_dict:
        obt_new = obt_new + jnp.einsum('jipq->ipq', param_dict['C'])

    if mc >= 3 and 'E' in param_dict:
        obt_new = obt_new + jnp.einsum('jkipq->ipq', param_dict['E'])

    # =========================================================================
    # 2-Body Space (using tbt_grouped)
    # =========================================================================
    if mc >= 2 and tbt_grouped is not None:
        if 'B' in param_dict:
            def inject_B(g_ij, B_ij):
                p = jnp.arange(nmodals)[:, None]
                r = jnp.arange(nmodals)[None, :]
                return g_ij.at[p, p, r, r].add(-B_ij)
            tbt_grouped = jax.vmap(jax.vmap(inject_B))(tbt_grouped, param_dict['B'])

        if 'C' in param_dict:
            def inject_C1(g_ij, C_ij):
                p = jnp.arange(nmodals)
                # Expand dims to broadcast the parameter correctly over the p slice
                return g_ij.at[p, p, :, :].add(-0.5 * jnp.expand_dims(C_ij, 0))
            
            def inject_C2(g_ij, C_ji):
                r = jnp.arange(nmodals)
                return g_ij.at[:, :, r, r].add(-0.5 * jnp.expand_dims(C_ji, -1))

            C = param_dict['C']
            C_ji = C.transpose(1, 0, 2, 3)
            tbt_grouped = jax.vmap(jax.vmap(inject_C1))(tbt_grouped, C)
            tbt_grouped = jax.vmap(jax.vmap(inject_C2))(tbt_grouped, C_ji)

    # =========================================================================
    # 3-Body Space (using trbt_grouped)
    # =========================================================================
    if mc >= 3 and trbt_grouped is not None:
        if 'D' in param_dict:
            def inject_D(v_ijk, D_ijk):
                p = jnp.arange(nmodals)[:, None, None]
                r = jnp.arange(nmodals)[None, :, None]
                t = jnp.arange(nmodals)[None, None, :]
                return v_ijk.at[p, p, r, r, t, t].add(-D_ijk)
            trbt_grouped = jax.vmap(jax.vmap(jax.vmap(inject_D)))(trbt_grouped, param_dict['D'])

        if 'E' in param_dict:
            def inject_E1(v_ijk, E_ijk):
                p = jnp.arange(nmodals)[:, None]
                r = jnp.arange(nmodals)[None, :]
                return v_ijk.at[p, p, r, r, :, :].add(-jnp.expand_dims(E_ijk, (0, 1)) / 3.0)
            
            def inject_E2(v_ijk, E_ikj):
                p = jnp.arange(nmodals)[:, None]
                t = jnp.arange(nmodals)[None, :]
                # JAX pushes separated advanced indices to the front, creating a (p, t, r, s) shape.
                return v_ijk.at[p, p, :, :, t, t].add(-jnp.expand_dims(E_ikj, (0, 1)) / 3.0)
            
            def inject_E3(v_ijk, E_kji):
                r = jnp.arange(nmodals)[:, None]
                t = jnp.arange(nmodals)[None, :]
                return v_ijk.at[:, :, r, r, t, t].add(-jnp.expand_dims(E_kji, (2, 3)) / 3.0)

            E = param_dict['E']
            E_ikj = E.transpose(0, 2, 1, 3, 4)
            E_kji = E.transpose(2, 1, 0, 3, 4)
            
            trbt_grouped = jax.vmap(jax.vmap(jax.vmap(inject_E1)))(trbt_grouped, E)
            trbt_grouped = jax.vmap(jax.vmap(jax.vmap(inject_E2)))(trbt_grouped, E_ikj)
            trbt_grouped = jax.vmap(jax.vmap(jax.vmap(inject_E3)))(trbt_grouped, E_kji)

        if 'F' in param_dict:
            def inject_F1(v_ijk, F_ijk):
                p = jnp.arange(nmodals)
                return v_ijk.at[p, p, :, :, :, :].add(-jnp.expand_dims(F_ijk, 0) / 3.0)
            
            def inject_F2(v_ijk, F_jik):
                r = jnp.arange(nmodals)
                return v_ijk.at[:, :, r, r, :, :].add(-jnp.expand_dims(F_jik, 2) / 3.0)
            
            def inject_F3(v_ijk, F_kji):
                t = jnp.arange(nmodals)
                F_kji_aligned = jnp.expand_dims(F_kji.transpose(2, 3, 0, 1), 4)
                return v_ijk.at[:, :, :, :, t, t].add(-F_kji_aligned / 3.0)

            F = param_dict['F']
            F_jik = F.transpose(1, 0, 2, 3, 4, 5, 6)
            F_kji = F.transpose(2, 1, 0, 3, 4, 5, 6)
            
            trbt_grouped = jax.vmap(jax.vmap(jax.vmap(inject_F1)))(trbt_grouped, F)
            trbt_grouped = jax.vmap(jax.vmap(jax.vmap(inject_F2)))(trbt_grouped, F_jik)
            trbt_grouped = jax.vmap(jax.vmap(jax.vmap(inject_F3)))(trbt_grouped, F_kji)

    # =========================================================================
    # Dense Fold-downs and Interleaving Reversion
    # =========================================================================
    tbt_new = None
    if tbt_grouped is not None:
        # i, j, p, q, r, s  ->  i, p, q, j, r, s
        tbt_new = tbt_grouped.transpose(0, 2, 3, 1, 4, 5)
        if mc >= 3 and 'F' in param_dict:
            tbt_new = tbt_new + jnp.einsum('kijpqrs->ipqjrs', param_dict['F'])

    trbt_new = None
    if trbt_grouped is not None:
        # i, j, k, p, q, r, s, t, u  ->  i, p, q, j, r, s, k, t, u
        trbt_new = trbt_grouped.transpose(0, 3, 4, 1, 5, 6, 2, 7, 8)

    return c_new, obt_new, tbt_new, trbt_new







def get_opt_Pauli_LCU_tensors_jax(C=None, obt=None, tbt=None, trbt=None):
    """JAX native version of the optimal LCU reduction."""
    C_tilde = 0.0 if C is None else C
    
    if obt is not None:
        C_tilde += jnp.einsum('ipp->', obt)/2.0 
    obt_tilde = obt
    if tbt is not None:
        C_tilde += jnp.einsum('ippjrr->', tbt)/4.0
        obt_tilde = obt_tilde + jnp.einsum('ipqjrr->ipq', tbt) if obt_tilde is not None else None
    tbt_tilde = tbt

    if trbt is not None:
        C_tilde += jnp.einsum('ippjrrktt->', trbt)/8.0
        obt_tilde = obt_tilde + jnp.einsum('ipqjrrktt->ipq', trbt) * 3.0/4.0 if obt_tilde is not None else None
        tbt_tilde = tbt_tilde + jnp.einsum('ipqjrsktt->ipqjrs', trbt) * 3.0/2.0 if tbt_tilde is not None else None
        trbt_tilde = trbt
    else:
        trbt_tilde = None

    return C_tilde, obt_tilde, tbt_tilde, trbt_tilde

def opt_Pauli_LCU_1norm_jax(obt, tbt, trbt=None):
    """JAX differentiable cost function to evaluate the 1-norm."""
    _, obt_tilde, tbt_tilde, trbt_tilde = get_opt_Pauli_LCU_tensors_jax(C=0.0, obt=obt, tbt=tbt, trbt=trbt)
    
    one_norm = 0
    if obt is not None:
        one_norm += jnp.sum(jnp.abs(obt_tilde))/2.0 
    if tbt is not None:
        one_norm += jnp.sum(jnp.abs(tbt_tilde))/4.0
    if trbt_tilde is not None:
        one_norm += jnp.sum(jnp.abs(trbt_tilde))/8.0
        
    return one_norm

# =============================================================================
# The High-Performance Optax API
# =============================================================================

def optimize_bliss_pauli(obt=None, tbt=None, trbt=None, initial_guess=None, syms=None | list, 
                    maxiter=1000, learning_rate=1e-3, chunk_size=100, ret_params=False):
    """
    Main entry point for minimizing the 1-norm of the Pauli LCU via Optax.
    
    Parameters:
    -----------
    obt : np.ndarray, optional
        1-body tensor (h).
    tbt : np.ndarray, optional
        2-body tensor (g). Assumed present for any mode-coupling calculation.
    trbt : np.ndarray, optional
        3-body tensor (v). If None, all 3-body parameters (D, E, F) are strictly 
        pruned from execution via dead code elimination.
    initial_guess : dict, np.ndarray, optional
        Initial parameters. Can be a full dictionary or minimal 1D array.
    syms : list of strings, optional
        List of symmetries to be included in the Killer operator. 
        It could be one of the non-empty subset of ['A', 'B', 'C', 'D', 'E', 'F'].
    maxiter : int
        Maximum iterations for the Adam optimizer.
    learning_rate : float
        Optimizer learning rate.
    chunk_size : int
        Logging interval for printing the loss value.
    ret_params : bool
        If True, the mathematically reconstructed dense parameter dict is returned.
        
    Returns:
    --------
    c_opt : float
    obt_opt : np.ndarray
    tbt_opt : np.ndarray or None
    trbt_opt : np.ndarray or None
    opt_params_dict : dict (only if ret_params=True)
    """
    mc = 3 if trbt is not None else (2 if tbt is not None else (1 if obt is not None else 0))

    if mc == 0:
        raise ValueError("At least one of obt, tbt, or trbt must be provided.")
    
    # Ensuring unwanted symmetries are removed
    if syms is not None:
        if mc < 3:
            if 'D' in syms: syms.remove('D')
            if 'E' in syms: syms.remove('E')
            if 'F' in syms: syms.remove('F')
        if tbt is None:
            if 'B' in syms: syms.remove('B')
            if 'C' in syms: syms.remove('C')
        if obt is None:
            if 'A' in syms: syms.remove('A')

    if obt is not None:
        nmodes, nmodals = obt.shape[0], obt.shape[-1]
    elif tbt is not None:
        nmodes, nmodals = tbt.shape[0], tbt.shape[-1]
    elif trbt is not None:
        nmodes, nmodals = trbt.shape[0], trbt.shape[-1]


    # Initialization and Static Determinations
    if isinstance(initial_guess, dict):
        param_keys = list(initial_guess.keys())
        scatter_mappings = build_scatter_indices(nmodes, nmodals, param_keys)
        params_1d = jnp.array(pack_minimal_bliss_params(initial_guess, scatter_mappings))
    elif initial_guess is None:
        if syms is None:
            if mc == 1:
                param_keys = ['A']
            elif mc == 2:
                param_keys = ['A', 'B', 'C']
            elif mc == 3:
                param_keys = ['A', 'B', 'C', 'D', 'E', 'F']
        else:
            param_keys = syms
        scatter_mappings = build_scatter_indices(nmodes, nmodals, param_keys)
        total_minimal_params = sum(num for _, num in scatter_mappings.values())
        params_1d = jnp.array(np.random.randn(total_minimal_params) * 0.1)
    
    if len(param_keys) == 0:
        raise ValueError("Provided tensors and list of symmetries are incompatible.")
    print ("Symmetries considered for BLISS:", param_keys)


    # Cast Hamiltonian inputs to immutable JAX arrays
    j_obt = jnp.array(obt) if obt is not None else None
    j_tbt = jnp.array(tbt) if tbt is not None else None
    j_trbt = jnp.array(trbt) if trbt is not None else None
        
    # Define JIT-compiled Loss and Step Functions
    @jax.jit
    def loss_fn(p_1d):
        # Unpack 1D array into mathematically rigid dense symmetric tensors
        dense_dict = unpack_minimal_bliss_params_jax(p_1d, scatter_mappings)
        
        # Compute Transform
        _, obt_new, tbt_new, trbt_new = get_bliss_hamiltonian_jax(nmodes, nmodals, j_obt, j_tbt, j_trbt, dense_dict, mc)
        
        # Compute the differentiable LCU 1-norm target
        return opt_Pauli_LCU_1norm_jax(obt_new, tbt_new, trbt_new)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params_1d)
    
    @jax.jit
    def step(p_1d, state):
        loss_val, grads = jax.value_and_grad(loss_fn)(p_1d)
        updates, state = optimizer.update(grads, state, p_1d)
        p_1d = optax.apply_updates(p_1d, updates)
        return p_1d, state, loss_val

    # Execute the Optimization Loop
    print(f"Starting optimization in {mc}-body space | "
          f"Parameters strictly tracking {total_minimal_params} minimal degrees of freedom.")
    
    for i in range(maxiter):
        params_1d, opt_state, loss_val = step(params_1d, opt_state)
        if i % chunk_size == 0 or i == maxiter - 1:
            print(f"Iteration {i:04d} | 1-Norm Loss = {loss_val:.6f}")

    # Final Hamiltonian Transformation & Type Coercion to standard NumPy
    final_dense_dict = unpack_minimal_bliss_params_jax(params_1d, scatter_mappings)
    c_opt, obt_opt, tbt_opt, trbt_opt = get_bliss_hamiltonian_jax(nmodes, nmodals, j_obt, j_tbt, j_trbt, final_dense_dict, mc)
    
    # Convert back to host memory standard NumPy arrays
    c_opt = float(c_opt)
    obt_opt = np.array(obt_opt)
    tbt_opt = np.array(tbt_opt) if tbt_opt is not None else None
    trbt_opt = np.array(trbt_opt) if trbt_opt is not None else None
    
    if ret_params:
        np_params_dict = {k: np.array(v) for k, v in final_dense_dict.items()}
        return c_opt, obt_opt, tbt_opt, trbt_opt, np_params_dict
    
    return c_opt, obt_opt, tbt_opt, trbt_opt















def optimize_bliss_sf(obt=None, tbt=np.ndarray, trbt=None, initial_guess=None, syms=None | list, 
                    maxiter=1000, learning_rate=1e-3, chunk_size=100, ret_params=False):
    """
    Main entry point for minimizing the Shatten norm of the single factorized two-body tensor via Optax.
    
    Parameters:
    -----------
    obt : np.ndarray, optional
        1-body tensor (h).
    tbt : np.ndarray, optional
        2-body tensor (g). Assumed present for any mode-coupling calculation.
    trbt : np.ndarray, optional
        3-body tensor (v). If None, all 3-body parameters (D, E, F) are strictly 
        pruned from execution via dead code elimination.
    initial_guess : dict, np.ndarray, optional
        Initial parameters. Can be a full dictionary or minimal 1D array.
    syms : list of strings, optional
        List of symmetries to be included in the Killer operator. 
        It could be one of the non-empty subsets of ['A', 'B', 'C', 'D', 'E', 'F'].
    maxiter : int
        Maximum iterations for the Adam optimizer.
    learning_rate : float
        Optimizer learning rate.
    chunk_size : int
        Logging interval for printing the loss value.
    ret_params : bool
        If True, the mathematically reconstructed dense parameter dict is returned.
        
    Returns:
    --------
    c_opt : float
    obt_opt : np.ndarray
    tbt_opt : np.ndarray or None
    trbt_opt : np.ndarray or None
    opt_params_dict : dict (only if ret_params=True)
    """
    # Initialization and Static Determinations
    nmodes, nmodals, _ = obt.shape
    mc = 3 if trbt is not None else (2 if tbt is not None else (1 if obt is not None else 0))

    if mc == 0:
        raise ValueError("At least one of obt, tbt, or trbt must be provided.")
    
    # Ensuring unwanted symmetries are removed
    if syms is not None:
        if mc < 3:
            if 'D' in syms: syms.remove('D')
            if 'E' in syms: syms.remove('E')
            if 'F' in syms: syms.remove('F')
        if tbt is None:
            if 'B' in syms: syms.remove('B')
            if 'C' in syms: syms.remove('C')
        if obt is None:
            if 'A' in syms: syms.remove('A')

    if obt is not None:
        nmodes, nmodals = obt.shape[0], obt.shape[-1]
    elif tbt is not None:
        nmodes, nmodals = tbt.shape[0], tbt.shape[-1]
    elif trbt is not None:
        nmodes, nmodals = trbt.shape[0], trbt.shape[-1]


    # Setting up the initial_guess
    if isinstance(initial_guess, dict):
        param_keys = list(initial_guess.keys())
        scatter_mappings = build_scatter_indices(nmodes, nmodals, param_keys)
        params_1d = jnp.array(pack_minimal_bliss_params(initial_guess, scatter_mappings))
    elif initial_guess is None:
        if syms is None:
            if mc == 1:
                param_keys = ['A']
            elif mc == 2:
                param_keys = ['A', 'B', 'C']
            elif mc == 3:
                param_keys = ['A', 'B', 'C', 'D', 'E', 'F']
        else:
            param_keys = syms
        scatter_mappings = build_scatter_indices(nmodes, nmodals, param_keys)
        total_minimal_params = sum(num for _, num in scatter_mappings.values())
        params_1d = jnp.array(np.random.randn(total_minimal_params) * 0.1)
    print ("Symmetries considered for BLISS:", param_keys)

    # Cast Hamiltonian inputs to immutable JAX arrays
    j_obt = jnp.array(obt) if obt is not None else None
    j_tbt = jnp.array(tbt) if tbt is not None else None
    j_trbt = jnp.array(trbt) if trbt is not None else None
        
    # Define JIT-compiled Loss and Step Functions
    @jax.jit
    def loss_fn(p_1d):
        # Unpack 1D array into mathematically rigid dense symmetric tensors
        dense_dict = unpack_minimal_bliss_params_jax(p_1d, scatter_mappings)
        
        # Compute Transform
        _, obt_new, tbt_new, trbt_new = get_bliss_hamiltonian_jax(nmodes, nmodals, j_obt, j_tbt, j_trbt, dense_dict, mc)
        
        # Compute the Shatten norm of the two-body tensor
        nmodes, nmodals = tbt_new.shape[0], tbt_new.shape[-1]
        tbt_new_mat = tbt_new.reshape(nmodes*nmodals**2, nmodes*nmodals**2)
        cost = jnp.sum(jnp.abs(jnp.linalg.eigvalsh(tbt_new_mat)))

        return cost

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params_1d)
    
    @jax.jit
    def step(p_1d, state):
        loss_val, grads = jax.value_and_grad(loss_fn)(p_1d)
        updates, state = optimizer.update(grads, state, p_1d)
        p_1d = optax.apply_updates(p_1d, updates)
        return p_1d, state, loss_val

    # Execute the Optimization Loop
    print(f"Starting optimization in {mc}-body space | "
          f"Parameters strictly tracking {total_minimal_params} minimal degrees of freedom.")
    
    for i in range(maxiter):
        params_1d, opt_state, loss_val = step(params_1d, opt_state)
        if i % chunk_size == 0 or i == maxiter - 1:
            print(f"Iteration {i:04d} | 1-Norm Loss = {loss_val:.6f}")

    # Final Hamiltonian Transformation & Type Coercion to standard NumPy
    final_dense_dict = unpack_minimal_bliss_params_jax(params_1d, scatter_mappings)
    c_opt, obt_opt, tbt_opt, trbt_opt = get_bliss_hamiltonian_jax(nmodes, nmodals, j_obt, j_tbt, j_trbt, final_dense_dict, mc)
    
    # Convert back to host memory standard NumPy arrays
    c_opt = float(c_opt)
    obt_opt = np.array(obt_opt)
    tbt_opt = np.array(tbt_opt) if tbt_opt is not None else None
    trbt_opt = np.array(trbt_opt) if trbt_opt is not None else None
    
    if ret_params:
        np_params_dict = {k: np.array(v) for k, v in final_dense_dict.items()}
        return c_opt, obt_opt, tbt_opt, trbt_opt, np_params_dict
    
    return c_opt, obt_opt, tbt_opt, trbt_opt