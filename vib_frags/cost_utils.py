# A module to estimate quantum resources for different fragmentation schemes
import numpy as np
from . import tensor_utils as tu
from copy import copy






def get_pruned_tensors_w_err(H1, H2_nonsym=None, H3_nonsym=None, MC=2, cutoff=0):
    
    # Process H1 using a memory-efficient mask
    mask1 = np.abs(H1) <= cutoff
    H1_approx = H1.copy()
    H1_approx[mask1] = 0
    err1 = np.sqrt(np.sum(H1[mask1]**2))
    
    # Process H2 using the same mask logic if it is provided
    err2 = 0.0
    H2_nonsym_approx = None
    if H2_nonsym is not None:
        mask2 = np.abs(H2_nonsym) <= cutoff
        H2_nonsym_approx = H2_nonsym.copy()
        H2_nonsym_approx[mask2] = 0
        err2 = np.sqrt(np.sum(H2_nonsym[mask2]**2))
        
    sparse_ten_error = err1 + err2
    
    H3_nonsym_approx = None
    if MC == 3 and H3_nonsym is not None:
        # Pre-allocate the output array for H3 to avoid massive copies
        H3_nonsym_approx = np.empty_like(H3_nonsym)
        err3_sq = 0.0
        
        # Process H3 slice-by-slice along the first dimension (chunking)
        for i in range(H3_nonsym.shape[0]):
            slice_h3 = H3_nonsym[i]
            mask3 = np.abs(slice_h3) <= cutoff
            
            # Accumulate the squared error strictly for pruned elements
            err3_sq += np.sum(slice_h3[mask3]**2)
            
            # Create the approximated slice
            approx_slice = slice_h3.copy()
            approx_slice[mask3] = 0
            
            # Place the finalized slice into the pre-allocated array
            H3_nonsym_approx[i] = approx_slice
            
        sparse_ten_error += np.sqrt(err3_sq)
        
    return H1_approx, H2_nonsym_approx, H3_nonsym_approx, sparse_ten_error




#Obtaining masked tensors for counting S in Pauli LCU
def mask_tensor_by_index_order(arr, axis_pairs):
    """
    arr : numpy array
    axis_pairs : list of tuples [(ax_x, ax_y), ...] where ax_x and ax_y are axes
                 indices in arr that should satisfy index_x >= index_y
    returns: arr with entries zeroed where ANY pair condition is False
    """
    final_mask = None
    for ax_x, ax_y in axis_pairs:
        dim_x = arr.shape[ax_x]
        dim_y = arr.shape[ax_y]
        idx_x = np.arange(dim_x)[:, None]    # shape (dim_x, 1)
        idx_y = np.arange(dim_y)[None, :]    # shape (1, dim_y)
        pair_mask = (idx_x >= idx_y)        # shape (dim_x, dim_y)

        # build shape of ones and insert pair_mask at the two axes
        shape = [1] * arr.ndim
        shape[ax_x] = dim_x
        shape[ax_y] = dim_y
        pair_mask_reshaped = pair_mask.reshape(shape)

        if final_mask is None:
            final_mask = pair_mask_reshaped
        else:
            final_mask = final_mask & pair_mask_reshaped

    # if no pairs given, return arr unchanged
    if final_mask is None:
        return arr
    # broadcast final_mask to arr shape and multiply
    return arr * final_mask



def get_Pauli_S(H1, H2_nonsym = None, H3_nonsym = None, cutoff = 0):
  """
  Finds the number of terms in the unary iteration in Pauli PREP circuit.
  It is important to note that the two and three body tensors need to be provided in the unsymmetrized form with shapes,
  H2_nonsym : np.array((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
  H3_nonsym : np.array((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
  """
  H1_masked = mask_tensor_by_index_order(H1, [(1,2)])               # p>=q
  S = np.sum(np.abs(H1_masked) > cutoff)
  if H2_nonsym is not None:
    H2_nonsym_masked = mask_tensor_by_index_order(H2_nonsym, [(1,2),(4,5)])         # p>=q and r>=s
    S += np.sum(np.abs(H2_nonsym_masked) > cutoff)
  if H3_nonsym is not None:
    H3_nonsym_masked = mask_tensor_by_index_order(H3_nonsym, [(1,2),(4,5),(7,8)])   # p>=q, r>=s, t>=u
    S += int(np.sum(np.abs(H3_nonsym_masked) > cutoff))
  return int(S)




def find_dirty_QROAM_k(S, m, dQ):
    """
    Find the optimal value of k in SelectSwap (QROAM) when the number of dirty ancillas, dQ, is given.
    """
    best_k = 1
    # Evaluate the baseline objective for k=1 to prevent blindly overwriting it
    min_objective = 2 * S
    
    k = 2
    # The loop relies on the physical constraints and the U-shape break condition
    while (k < S) and (k <= (dQ / m) + 1):        
        # Evaluate the objective function
        objective_val = 2 * np.ceil(S / k) + 4 * m * (k - 1)
        
        # Since the function is strictly U-shaped, if the objective goes up, we have passed the minimum
        if objective_val > min_objective:
            break
            
        # Update the best known values
        if objective_val < min_objective:
            min_objective = objective_val
            best_k = k
            
        # Move to the next power of 2
        k *= 2
        
    return int(best_k)


def find_clean_QROAM_k(S, m, cQ):
    """
    Find the optimal value of k in SelectSwap (QROAM) when the number of clean ancillas, cQ, is given.
    Note that if cQ is too small, the default value of k = 1 is returned.
    """
    best_k = 1
    # Evaluate the baseline objective for k=1
    min_objective = S
    
    k = 2
    while k < S:
        # Evaluate the inequality constraint using the new variables
        constraint_val = (k - 1) * m + np.ceil(np.log2(S / k))
        
        # Early Stop: If the constraint exceeds cQ, it will only get worse for larger k
        if constraint_val > cQ:
            break
            
        # Evaluate the objective function
        objective_val = np.ceil(S / k) + m * (k - 1)
        
        # Early Stop: If the objective went up, we already passed the minimum
        if objective_val > min_objective:
            break
            
        # Update the best known values
        if objective_val < min_objective:
            min_objective = objective_val
            best_k = k
            
        # Move to the next power of 2
        k *= 2
        
    return int(best_k)




def get_QROAM_cost(S, m, dirty=False, clean=False, nmodes=None, nmodals=None):
    """
    Calculates the Toffoli and ancilla costs for QROAM data loading.

    Evaluates the resource requirements given the size of the index register (S)
    and the QROM volume being loaded (m). The function optimizes multiplexing
    parameters based on the availability of clean or dirty ancilla qubits.

    Note:
        The returned ancilla costs exclude the index register (cost: log2(S))
        and the output container (cost: m). The baseline clean ancilla requirement
        is bounded by log2(S); deficits in the provided budget will default
        to this minimum.

    Args:
        S (int): Size of the index register.
        m (int): Volume of the QROM being loaded.
        dirty (int | bool): Available dirty ancillas, or flag to calculate via modes/modals.
        clean (int | bool): Available clean ancillas, or flag assuming optimal unconstrained availability.
        nmodes (int, optional): Number of modes. Required if dirty is True.
        nmodals (int, optional): Number of modals. Required if dirty is True.

    Returns:
        tuple: A tuple containing (Toffoli_cost, clean_cost, dirty_cost) as integers.
    """
    
    # Process the dirty ancilla case if requested
    if dirty:
        if isinstance(dirty, bool):
            if nmodes is None or nmodals is None:
                raise ValueError("Explicit values for nmodes and nmodals are required when 'dirty' is passed as a boolean.")
            dQ = nmodes * nmodals
        else:
            dQ = dirty
            
        # Supplement the dirty budget with any available clean qubits
        dQ += int(clean)
        
        # Revert to standard QROM scaling if the available dirty qubits fall below the required threshold
        if dQ < 2 * m:
            print(f"Warning: Insufficient dirty qubits ({dQ}). QROAM requires at least twice the volume ({2*m}). Defaulting to standard QROM.")
            Toffoli_cost = S - 1
            clean_cost = np.ceil(np.log2(S))
            return int(Toffoli_cost), int(clean_cost), 0
            
        # Determine the optimal parameter and calculate associated costs
        k = find_dirty_QROAM_k(S, m, dQ)
        Toffoli_cost = 2 * np.ceil(S / k) + 4 * m * (k - 1)
        clean_cost = np.ceil(np.log2(S / k))
        dirty_cost = (k - 1) * m
        
        # Shift any surplus dirty cost to the clean budget if physical bounds are exceeded
        available_dirty = dQ - int(clean)
        if dirty_cost > available_dirty:
            clean_cost += (dirty_cost - available_dirty)
            dirty_cost = available_dirty
            
        return int(Toffoli_cost), int(clean_cost), int(dirty_cost)

    # Process the unconstrained optimal clean ancilla case
    if clean is True:
        print("Notice: Assuming optimal unconstrained clean ancilla availability for QROAM.")
        limit = int(np.floor(np.sqrt(S / m)))
        
        # Snap the limit to the nearest lower power of 2
        k = 1 << (limit.bit_length() - 1)
        
        Toffoli_cost = np.ceil(S / k) + m * (k - 1)
        clean_cost = np.ceil(np.log2(S / k)) + (k - 1) * m
        return int(Toffoli_cost), int(clean_cost), 0

    # Process the constrained clean ancilla case based on a strict budget
    elif type(clean) is int:
        k = find_clean_QROAM_k(S, m, clean)
        Toffoli_cost = np.ceil(S / k) + m * (k - 1)
        clean_cost = np.ceil(np.log2(S / k)) + (k - 1) * m
        return int(Toffoli_cost), int(clean_cost), 0

    # Fallback to standard QROM scaling if no valid optimization parameters are provided
    else:
        Toffoli_cost = S - 1
        clean_cost = np.ceil(np.log2(S))
        return int(Toffoli_cost), int(clean_cost), 0






def find_dirty_QROAM_inv_k(S, dQ):
    """
    Find the optimal parameter k for inverse QROAM using dirty ancillas.
    """
    best_k = 1
    # Evaluate the baseline objective for k=1 to prevent blindly overwriting it
    min_objective = 2 * S + 4
    
    k = 2
    # The loop relies on the physical dirty qubit constraints and the U-shape break condition
    while (k < S) and (k <= dQ + 1):
        # Evaluate the Toffoli cost objective function
        objective_val = 2 * np.ceil(S / k) + 4 * k
        
        # Break early if the U-shaped objective function begins to increase
        if objective_val > min_objective:
            break  
            
        # Update the best known values upon finding a lower cost
        if objective_val < min_objective:
            min_objective = objective_val
            best_k = k
            
        # Move to the next power of 2
        k *= 2
        
    return int(best_k)


def find_clean_QROAM_inv_k(S, cQ):
    """
    Find the optimal parameter k for inverse QROAM using clean ancillas.
    """
    best_k = 1
    # Evaluate the baseline objective for k=1
    min_objective = S + 1
    
    k = 2
    while k < S:
        # Evaluate the clean qubit constraint
        constraint_val = np.ceil(np.log2(S / k)) + k
        
        # Break early if the required clean qubits exceed the available budget
        if constraint_val > cQ:
            break
            
        # Evaluate the Toffoli cost objective function
        objective_val = np.ceil(S / k) + k
        
        # Break early if the U-shaped objective function begins to increase
        if objective_val > min_objective:
            break
            
        # Update the best known values upon finding a lower cost
        if objective_val < min_objective:
            min_objective = objective_val
            best_k = k
            
        # Move to the next power of 2
        k *= 2
        
    return int(best_k)


def get_QROAM_inv_cost(S, dirty=False, clean=False, nmodes=None, nmodals=None):
    """
    Calculates the Toffoli and ancilla costs for inverse QROAM based on measurement-based uncomputation.

    Evaluates the resource requirements given the size of the index register (S).
    The function optimizes parameters based on the availability of clean or dirty ancilla qubits.

    Args:
        S (int): Size of the index register.
        dirty (int | bool): Available dirty ancillas, or flag to calculate via modes/modals.
        clean (int | bool): Available clean ancillas, or flag assuming optimal unconstrained availability.
        nmodes (int, optional): Number of modes. Required if dirty is True.
        nmodals (int, optional): Number of modals. Required if dirty is True.

    Returns:
        tuple: A tuple containing (Toffoli_cost, clean_cost, dirty_cost) as integers.
    """
    # Process the dirty ancilla case if requested
    if dirty:
        if isinstance(dirty, bool):
            if nmodes is None or nmodals is None:
                raise ValueError("Explicit values for nmodes and nmodals are required when 'dirty' is passed as a boolean.")
            dQ = nmodes * nmodals
        else:
            dQ = dirty
            
        # Supplement the dirty budget with any available clean qubits
        dQ += int(clean)
        
        # Determine the optimal parameter and calculate associated costs
        k = find_dirty_QROAM_inv_k(S, dQ)
        Toffoli_cost = 2 * np.ceil(S / k) + 4 * k
        clean_cost = np.ceil(np.log2(S / k)) + 1
        dirty_cost = k - 1
        
        return int(Toffoli_cost), int(clean_cost), int(dirty_cost)

    # Process the unconstrained optimal clean ancilla case
    if clean is True:
        print("Notice: Assuming optimal unconstrained clean ancilla availability for inverse QROAM.")
        limit = int(np.floor(np.sqrt(S)))
        
        # Snap the limit to the nearest lower power of 2
        k = 1 << (limit.bit_length() - 1)
        
        Toffoli_cost = np.ceil(S / k) + k
        clean_cost = np.ceil(np.log2(S / k)) + k
        return int(Toffoli_cost), int(clean_cost), 0

    # Process the constrained clean ancilla case based on a strict budget
    elif type(clean) is int:
        # Determine the optimal parameter within the clean budget
        k = find_clean_QROAM_inv_k(S, clean)
        Toffoli_cost = np.ceil(S / k) + k
        clean_cost = np.ceil(np.log2(S / k)) + k
        return int(Toffoli_cost), int(clean_cost), 0

    # Fallback to standard QROM scaling if no valid optimization parameters are provided
    else:
        Toffoli_cost = S - 1
        clean_cost = np.ceil(np.log2(S))
        return int(Toffoli_cost), int(clean_cost), 0





# Toffoli and qubit cost of Pauli PREP circuit and its inverse
def get_Pauli_PREP_UNPREP_cost(nmodes, nmodals, S, aleph, MC, br = 20, dirty = False, clean = False, verbose = False):
  """
  S can be obtained using get_Pauli_S function. aleph stands for the number of bits precicion used in expressing LCU coefficients.
  br is the number of bits used to specify rotation angle in generating uniform superposition state.
  """
  #Some constants:
  nM = np.ceil(np.log2(nmodes))
  nN = np.ceil(np.log2(nmodals))
  nMC = np.ceil(np.log2(MC))
  eta = (S & -S).bit_length() - 1                         #eta is the largest integer such that 2^eta is a factor of S. Used in uniform superposition
  m = 2*MC*(nM + 2*nN + 1) + 2*nMC + 2 + aleph                 #size of the register loaded by QROM

  uniform_state_cost = 3*np.ceil(np.log2(S)) - 3*eta + 2*br - 9

  QROAM_cost, QROAM_clean, QROAM_dirty = get_QROAM_cost(S, m, dirty, clean, nmodes, nmodals)
  QROAM_inv_cost, QROAM_inv_clean, QROAM_inv_dirty = get_QROAM_inv_cost(S, dirty, QROAM_clean, nmodes, nmodals)
  
  if verbose:
    print (f"\nQROAM ancilla cost: {QROAM_clean}")
    print (f"QROAM inverse ancilla cost: {QROAM_inv_clean}")

  #coherent alias sampling (cas) cost
  cas_cost = aleph

  #index <-> alt swap cost controlled on output of cas
  swp_cost = MC*(nM + 2*nN + 1) + nMC + 1

  # PREP cost
  C_P = uniform_state_cost + QROAM_cost + cas_cost + swp_cost
  # UNPREP cost
  C_Pdag = uniform_state_cost + QROAM_inv_cost + cas_cost + swp_cost
  # Qubit cost
  C_P_qubits = np.ceil(np.log2(S)) + 2 + br + m + aleph + 1 + MC + QROAM_clean

  if QROAM_inv_clean > QROAM_clean:
    C_P_qubits += int(QROAM_inv_clean - QROAM_clean)


  return C_P, C_Pdag, C_P_qubits



# Toffoli and qubit cost of Pauli 
def get_Pauli_SELECT_cost(nmodes, nmodals, MC):
  #Cost of primitive
  C_SubSel = 2*nmodes*nmodals + 2*nmodes + 3*nmodals - 6
  #Total SELECT cost
  C_S = MC*C_SubSel + MC - 1           
  C_S_qubits = nmodes*nmodals
  return C_S, C_S_qubits




def get_Pauli_QPE_cost(one_norm, eps, nmodes, nmodals, S, MC, br = 20, aleph = 20, dirty = False, clean = False, verbose = True):  
  """
  Function to obtain the cost of performing QPE to a target accuracy of eps using Pauli LCU based block encoding.
  """
  if dirty == True:
    dirty = nmodes*nmodals + aleph + br                         #The system qubits, aleph qubits from alias sampling, and the ancillas from uniform superposition are idle dirty qubits

  # PREP and PREP inverse cost
  C_P, C_Pdag, C_P_qubits = get_Pauli_PREP_UNPREP_cost(nmodes, nmodals, S, aleph, MC, br, dirty, clean, verbose)
  # SELECT cost
  C_S, C_S_qubits = get_Pauli_SELECT_cost(nmodes, nmodals, MC)

  #Block encoding cost
  C_BE = C_P + C_S + C_Pdag
  # C_BE = C_P + C_S + C_P

  QPE_nsteps = np.ceil(np.pi*one_norm/(2*eps))           # Multiplier to Walker operator cost to get QPE cost

  #Total energy estimation cost
  C_Pauli = QPE_nsteps*(C_BE + np.ceil(np.log2(S)) + aleph + 4)
  N_Pauli = 2*np.ceil(np.log2(QPE_nsteps + 1)) - 1 + C_P_qubits + C_S_qubits

  if verbose:
    print (f"\nCost of PREP: {C_P: 0.2e}")
    print (f"Cost of SEL: {C_S: 0.2e}")
    print (f"Cost of BE: {C_BE: 0.2e}")
    print (f"LCU 1-norm: {one_norm: 0.2e} cm-1")
    print (f"# of walks: {QPE_nsteps: 0.2e}")
    print (f"Toffoli cost of Sparse Pauli LCU: {C_Pauli: 0.5e}")
    print (f"Qubit cost of Sparse Pauli LCU: {N_Pauli}")
  
  return C_Pauli, N_Pauli














#__________________________________________________________________________________________________________________
#THC cost utils
#__________________________________________________________________________________________________________________
def get_THC_BE_cost(nmodes, nmodals, S, R, nthc, MC, br = 20, aleph = 20, beth = 20, dirty = False, clean = False, verbose = False):
  """
  Function to obtain the Toffoli and qubit costs of block encoding a THC LCU
  """
  #PREP and UNPREP cost
  #Some constants:
  nM = np.ceil(np.log2(nmodes))
  nR = np.ceil(np.log2(R))
  nMC = np.ceil(np.log2(MC))
  #eta is the largest integer such that 2^eta is a factor of S. Used in uniform superposition
  eta = (S & -S).bit_length() - 1                         
  #Data loader output size
  m = 2*MC*(nM + nR) + 2*nMC + 2 + aleph

  #Cost of uniform superposition of S states
  uniform_state_cost = 3*np.ceil(np.log2(S)) - 3*eta + 2*br - 9

  PREP_QROAM_cost, PREP_QROAM_clean, _ = get_QROAM_cost(S, m, dirty, clean, nmodes, nmodals)
  UNPREP_QROAM_inv_cost, UNPREP_QROAM_inv_clean, _ = get_QROAM_inv_cost(S, dirty, PREP_QROAM_clean, nmodes, nmodals)
  
  if verbose:
    print (f"\nQROAM ancilla cost: {PREP_QROAM_clean}")
    print (f"QROAM inverse ancilla cost: {UNPREP_QROAM_inv_clean}")
  
  #coherent alias sampling (cas) cost
  cas_cost = aleph
  #index <-> alt swap cost controlled on output of cas
  swp_cost = MC*(nM + nR) + nMC + 1

  #Complete PREP cost
  C_P = uniform_state_cost + PREP_QROAM_cost + cas_cost + swp_cost
  #Complete UNPREP cost
  C_Pdag = uniform_state_cost + UNPREP_QROAM_inv_cost + cas_cost + swp_cost

  #SELECT cost
  C_S = MC - 1
  m_tilde = (nmodals - 1)*beth
  for l in range (MC):
     Sl = (MC - l) * sum(nthc[l:]) + (0 if l > 0 else nmodes * nmodals - sum(nthc))
     clean_new = max(PREP_QROAM_clean - m_tilde, beth - 2) if clean > 0 else False
     SEL_QROAM_cost, SEL_QROAM_clean, _ = get_QROAM_cost(Sl, m_tilde, False, clean_new, nmodes, nmodals)
     SEL_QROAM_inv_cost, SEL_QROAM_inv_clean, _ = get_QROAM_inv_cost(Sl, False, clean_new, nmodes, nmodals)
     C_SubSel_l = 2*(nmodes-l-1) + SEL_QROAM_cost + SEL_QROAM_inv_cost + 4*(nmodals - 1)*(beth - 2)
     C_S += C_SubSel_l

  #Total logical qubit cost
  N_BE = np.ceil(np.log2(S)) + max(br, beth) + 2 + m + aleph + 1 + nmodes*nmodals + max(PREP_QROAM_clean, max(SEL_QROAM_clean, beth - 2) + m_tilde)

  return C_P, C_S, C_Pdag, N_BE




#Cost of QPE of THC Hamiltonian
def get_THC_QPE_cost(lmbda, eps, nmodes, nmodals, nthc, MC, br = 20, aleph = 20, beth = 20, PREP_vol = None, dirty = False, clean = False, diff_angles = True, verbose = True):
  
  #Number of terms in LCU
  if type(nthc) is int:
    S = int(nmodes*nmodals + (nmodes*(nmodes - 1)/2)*nthc**2)        # Number of |ip> + number of |iu>|jv>
    if MC == 3:
      S += int((nmodes*(nmodes - 1)*(nmodes-2)/6)*nthc**3)           # Number of |iu>|jv>|kw>
    R = nthc
    nthc = [nthc,]*nmodes
  else:
    R_sum = 0
    for i in range (nmodes):
      for j in range (i):
        R_sum += nthc[i] * nthc[j]
    if MC == 3:
      for i in range (nmodes):
        for j in range (i):
          for k in range (j):
            R_sum += nthc[i] * nthc[j] * nthc[k]
    S = int(nmodes*nmodals + R_sum)
    R = max(nthc)

  if PREP_vol is not None:
    S = PREP_vol

  if dirty == True:
    dirty = nmodes*nmodals + aleph + nmodals*beth - 2                 #The system qubits, aleph qubits from alias sampling, and the ancillas for orbital rotation are idle dirty qubits


  # C_P, C_Pdag, C_P_qubits = get_THC_PREP_UNPREP_cost(nmodes, nmodals, S, R, nthc, MC, br, aleph, PREP_vol, dirty, clean, verbose)
  # C_S, C_S_qubits = get_THC_SEL_cost(nmodes, nmodals, nthc, MC, beth, subsel_a, diff_angles, verbose)


  #Number of steps in QPE i.e. number of access to walker operator
  QPE_nsteps = np.ceil(np.pi*lmbda/(2*eps))
  #Cost of single walker = Cost of block encoding + cost of controlled reflection
  C_P, C_S, C_Pdag, N_W = get_THC_BE_cost(nmodes, nmodals, S, R, nthc, MC, br, aleph, beth, dirty, clean, verbose)
  C_BE = C_P + C_S + C_Pdag
  C_W = C_BE + np.ceil(np.log2(S)) + aleph + 1

  #Total Toffoli cost of THC LCU
  C_THC = QPE_nsteps*C_W

  #Totoal qubit cost of THC LCU
  N_THC = int(2*np.ceil(np.log2(QPE_nsteps + 1)) - 1 + N_W)

  if verbose:
    print (f"\nCost of PREP: {C_P: 0.2e}")
    print (f"Cost of SEL: {C_S: 0.2e}")
    print (f"Cost of BE: {C_BE: 0.2e}")
    print (f"LCU 1-norm: {lmbda: 0.2e} cm-1")
    print (f"# of walks: {QPE_nsteps: 0.2e}")
    print (f"Toffoli cost of THC LCU: {C_THC: 0.5e}")
    print (f"Qubit cost of THC LCU: {N_THC}")
  
  return C_THC, N_THC

