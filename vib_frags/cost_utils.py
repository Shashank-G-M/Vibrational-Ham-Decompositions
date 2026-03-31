# A module to estimate quantum resources for different fragmentation schemes
import numpy as np
from . import tensor_utils as tu
import warnings
from copy import copy



#Obtain pruned tensors for sparse Pauli LCU along with error in approximation
def get_pruned_tensors_w_err(H1, H2_nonsym = None, H3_nonsym = None, MC = 2, cutoff = 0):
  """
  H1 : np.array((nmodes, nmodals, nmodals))
  H2 : np.array((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
  H3 : np.array((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
  """
  H1_approx = np.where(np.abs(H1) > cutoff, H1, 0)
  H2_nonsym_approx = np.where(np.abs(H2_nonsym) > cutoff, H2_nonsym, 0)
  H3_nonsym_approx = None
  sparse_ten_error = np.sqrt(np.sum((H1_approx - H1)**2)) + np.sqrt(np.sum((H2_nonsym_approx - H2_nonsym)**2))
  if MC == 3:
    _mask = np.abs(H3_nonsym) <= cutoff
    H3_nonsym_approx = copy(H3_nonsym)
    H3_nonsym_approx[_mask] = 0
    sparse_ten_error += np.sqrt(np.sum(H3_nonsym[_mask]**2))
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
  min_objective = float('inf')

  k = 2
  while ((k < S) and (k <= np.sqrt(2*S / m))) and (k <= dQ/m + 1):        
    # 2. Evaluate the objective function
    objective_val = 2 * np.ceil(S / k) + 4 * m * (k - 1)
    
    # Early Stop 2: Since the function is strictly U-shaped, if the 
    # objective goes up, we have passed the minimum.
    if objective_val > min_objective:
        break
        
    # 3. Update the best known values
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
  min_objective = float('inf')
  
  k = 2
  # The loop now requires k to be less than S AND \leq the optimal value of sqrt(S/m) 
  while k < S and k <= np.sqrt(S / m):
      # 1. Evaluate the inequality constraint using the new variables
      constraint_val = (k - 1) * m + np.ceil(np.log2(S / k))
      
      # Early Stop 1: If the constraint exceeds cQ, it will only get worse for larger k
      if constraint_val > cQ:
          break
          
      # 2. Evaluate the objective function
      objective_val = np.ceil(S / k) + m * (k - 1)
      
      # Early Stop 2: If the objective went up, we already passed the minimum
      if objective_val > min_objective:
          break
          
      # 3. Update the best known values
      if objective_val < min_objective:
          min_objective = objective_val
          best_k = k
          
      k *= 2
      
  return int(best_k)








def get_QROAM_cost(S, m, dirty=bool | int, clean=bool | int, nmodes=None, nmodals=None):
  """
  Get the general Toffoli and ancilla cost of QROAM when the size of the index register and the QROM volume being loaded is given.
  To use dirty ancillas, one must provide either the number of modes and modals, or the number of dirty qubits available.
  Note that the number of clean ancillas needed for QROAM is lower bounded by the number of clean ancillas needed for QROM, which is equal to
  log(S). So if the input parameter "clean" is smaller than this, the output clean_cost will be log(S).
  
  (Keep in mmind that the qubit cost here means just the ancilla cost. The index register itself has a qubit cost of log(S) and the output container has cost m.
  These values are not returned by this function.)
  """
  if dirty > 0:
    if type(dirty) is bool:
      if nmodes is None or nmodals is None:
        raise ValueError("nmodes and nmodals must be provided if number of dirty qubits is not explicitly provided")
      dQ = nmodes*nmodals     #Number of dirty qubits
    else:
      dQ = dirty
    dQ += int(clean)                       #If additional clean qubits are provided along with dirty qubits, then clean qubits will be treated as dirty.
    if dQ < 2*m:
      warnings.warn(f"The number of available dirty qubits {(dQ)} must not be less than twice the size of the QROM volume {(m)}. Switching to QROM.", UserWarning)
    else:
      # Calculate the optimal possible value of k
      k = find_dirty_QROAM_k(S, m, dQ)
      Toffoli_cost = 2*np.ceil(S/k) + 4*m*(k-1)
      clean_cost = np.ceil(np.log2(S/k))
      dirty_cost = (k-1)*m
      if dirty_cost > dQ - int(clean):
        clean_cost += dirty_cost - (dQ - int(clean))
        dirty_cost = dQ - int(clean)
      return int(Toffoli_cost), int(clean_cost), int(dirty_cost)
  
  if (type(clean) is bool) and (clean == True):
    print ("Assuming optimal number of clean ancillas for QROAM")
    limit = int(np.floor(np.sqrt(S/m)))
    k = 1 << (limit.bit_length() - 1)
    Toffoli_cost = np.ceil(S/k) + m*(k-1)
    clean_cost = np.ceil(np.log2(S/k)) + (k-1)*m
    return int(Toffoli_cost), int(clean_cost), int(0)
  elif type(clean) is int:
    cQ = clean
    # Calculate the optimal possible value of k
    k = find_clean_QROAM_k(S, m, cQ)
    Toffoli_cost = np.ceil(S/k) + m*(k-1)
    clean_cost = np.ceil(np.log2(S/k)) + (k-1)*m
    return int(Toffoli_cost), int(clean_cost), int(0)
  else:
    Toffoli_cost = S - 1
    clean_cost = np.ceil(np.log2(S))
    return int(Toffoli_cost), int(clean_cost), int(0)
    





def find_dirty_QROAM_inv_k(S, dQ):
  best_k = 1
  min_objective = float('inf')
  k = 2
  # All upper-bound constraints combined directly into the loop condition
  while k <= np.sqrt(S / 2) and k <= dQ + 1:
      # Evaluate the objective function
      objective_val = 2 * np.ceil(S / k) + 4 * k
      # Early Stop: If the objective goes up, we have passed the minimum
      if objective_val > min_objective:
          break  
      # Update the best known values
      if objective_val < min_objective:
          min_objective = objective_val
          best_k = k
      # Move to the next power of 2
      k *= 2
  return int(best_k)





def find_clean_QROAM_inv_k(S, cQ):
  best_k = 1
  min_objective = float('inf')
  
  k = 2
  # The condition k <= np.sqrt(S) naturally enforces k < S
  while k <= np.sqrt(S):
      
      # 1. Evaluate the constraint
      constraint_val = np.ceil(np.log2(S / k)) + k
      
      # Early Stop 1: constraint_val strictly increases as k doubles. 
      if constraint_val > cQ:
          break
          
      # 2. Evaluate the objective function
      objective_val = np.ceil(S / k) + k
      
      # Early Stop 2: If the objective goes up, we passed the minimum.
      if objective_val > min_objective:
          break
          
      # 3. Update the best known values
      if objective_val < min_objective:
          min_objective = objective_val
          best_k = k
          
      # Move to the next power of 2
      k *= 2
      
  return int(best_k)







def get_QROAM_inv_cost(S, dirty=bool | int, clean=bool | int, nmodes=None, nmodals=None):
  """
  Get the cost of inverse QROAM based on measurement based uncomputation.
  """
  if dirty > 0:
    if type(dirty) is bool:
      if (nmodes is None) or (nmodals is None):
        raise ValueError("nmodes and nmodals must be provided if number of dirty qubits is not explicitly provided")
      dQ = nmodes*nmodals     #Number of dirty qubits
    else:
      dQ = dirty
    dQ += int(clean)                       #If additional clean qubits are provided along with dirty qubits, then clean qubits will be treated as dirty.
    # Calculate the optimum possible value for k
    k = find_dirty_QROAM_inv_k(S, dQ)
    Toffoli_cost = 2*np.ceil(S/k) + 4*k
    clean_cost = np.ceil(np.log2(S/k)) + 1
    dirty_cost = k-1
    return int(Toffoli_cost), int(clean_cost), int(dirty_cost)

  if (type(clean) is bool) and (clean == True):
    limit = int(np.floor(np.sqrt(S)))
    k = 1 << (limit.bit_length() - 1)
    Toffoli_cost = np.ceil(S/k) + k
    clean_cost = np.ceil(np.log2(S/k)) + k
    return int(Toffoli_cost), int(clean_cost), int(0)
  elif type(clean) is int:
    cQ = clean
    # Calculate the optimum possible value for k
    k = find_clean_QROAM_inv_k(S, cQ)
    Toffoli_cost = np.ceil(S/k) + k
    clean_cost = np.ceil(np.log2(S/k)) + k
    return int(Toffoli_cost), int(clean_cost), int(0)
  else:
    Toffoli_cost = S - 1
    clean_cost = np.ceil(np.log2(S))
    return int(Toffoli_cost), int(clean_cost), int(0)







# Toffoli and qubit cost of Pauli PREP circuit and its inverse
def get_Pauli_PREP_UNPREP_cost(nmodes, nmodals, S, aleph, MC, br = 20, dirty = False, clean = False, verbose = False):
  """
  S can be obtained using get_Pauli_S function. aleph stands for the number of bits precicion used in expressing LCU coefficients.
  br is the number of bits used to specify rotation angle in generating uniform superposition state.
  """
  #Some constants:
  nM = np.ceil(np.log2(nmodes))
  nN = np.ceil(np.log2(nmodals))
  eta = (S & -S).bit_length() - 1                         #eta is the largest integer such that 2^eta is a factor of S. Used in uniform superposition
  m = 2*MC*(nM + 2*nN + 1) + 2*MC + aleph                 #size of the register loaded by QROM
  QROAM_cost, QROAM_clean, QROAM_dirty = get_QROAM_cost(S, m, dirty, clean, nmodes, nmodals)
  QROAM_inv_cost, QROAM_inv_clean, QROAM_inv_dirty = get_QROAM_inv_cost(S, dirty, clean, nmodes, nmodals)
  
  if verbose:
    print (f"\nQROAM ancilla cost: {QROAM_clean}")
    print (f"QROAM inverse ancilla cost: {QROAM_inv_clean}")

  uniform_state_cost = 3*np.ceil(np.log2(S)) - 3*eta + 2*br

  # PREP cost
  C_P = uniform_state_cost - 9 + QROAM_cost + aleph + MC*(nM + 2*nN + 1) + (MC-1) + MC*nN  
  # UNPREP cost
  C_Pdag = uniform_state_cost - 9 + QROAM_inv_cost                 #No additional UNPREP cost is mentioned in appendix of THC
  # Qubit cost
  C_P_qubits = br + 2 + np.ceil(np.log2(S)) + m + QROAM_clean + aleph + 1 + MC 

  if QROAM_inv_clean > QROAM_clean:
    C_P_qubits += int(QROAM_inv_clean - QROAM_clean)


  return C_P, C_Pdag, C_P_qubits



# Toffoli and qubit cost of Pauli 
def get_Pauli_SELECT_cost(nmodes, nmodals, MC):
  #Cost of primitive
  C_S = MC*(2*nmodes*nmodals + 2*nmodes + 3*nmodals - 6) + 2              
  C_S_qubits = 2*MC - 1 + nmodes*nmodals
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
  C_Pauli = QPE_nsteps*(C_BE + np.ceil(np.log2(S)) + aleph + 2)
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


#Cost of THC state preparation and inverse state preparation
def get_THC_PREP_UNPREP_cost(nmodes, nmodals, S, R, nthc, MC, br = 20, aleph = 20, PREP_vol = None, dirty = False, clean = False, verbose = False):
  """
  Calculate the Toffoli and qubit cost of state preparation and its inverse in block encoding a vibrational Hamiltonian expressed in terms of THC decomposition.

  Args:
    optimize: If True, performs optimal QROAM instead of QROM (default is False)
    br: number of bits used in expressing the angle for uniform state preparation (default is 20)
    aleph: number of bits used in approximating the LCU coeffcients (default is 20)
    inv_cost: If True, provides the cost of inverse state preparation i.e. PREP dagger
    PREP_vol: Size of the data to be loaded can be provided if already precomputed. Otherwise, will be computed without using any truncation
  """

  if verbose:
    print ("Number of terms in THC LCU = ", S)

  #eta is the largest integer such that 2^eta is a factor of S. Used in uniform superposition
  eta = (S & -S).bit_length() - 1                         

  #Cost of uniform superposition of S states
  uni_cost = 3*np.ceil(np.log2(S)) - 3*eta + 2*br - 9

  #Data loader output size
  m = 2*MC*(np.ceil(np.log2(nmodes)) + np.ceil(np.log2(R))) + aleph + 2 + 2*(np.ceil(np.log2(MC)))

  QROAM_cost, QROAM_clean, QROAM_dirty = get_QROAM_cost(S, m, dirty, clean, nmodes, nmodals)
  QROAM_inv_cost, QROAM_inv_clean, QROAM_inv_dirty = get_QROAM_inv_cost(S, dirty, clean, nmodes, nmodals)
  
  if verbose:
    print (f"\nQROAM ancilla cost: {QROAM_clean}")
    print (f"QROAM inverse ancilla cost: {QROAM_inv_clean}")
  
  #coherent alias sampling (cas) cost
  cas_cost = aleph

  #index <-> alt swap cost controlled on output of cas
  swp_cost = MC*(np.ceil(np.log2(nmodes)) + np.ceil(np.log2(R))) + np.ceil(np.log2(MC))

  #Complete PREP cost
  C_P = uni_cost + QROAM_cost + cas_cost + swp_cost

  #Complete UNPREP cost
  C_P_dag = uni_cost + QROAM_inv_cost + cas_cost + swp_cost

  #Qubit cost
  C_P_qubits =  br + 2 + np.ceil(np.log2(S)) + m + QROAM_clean + aleph + 1

  # Include any additional qubits that are required to uncompute QROAM. Usually this not necessary.
  if QROAM_inv_clean > QROAM_clean:
    C_P_qubits += int(QROAM_inv_clean - QROAM_clean)

  return C_P, C_P_dag, C_P_qubits




#Cost of THC select
def get_THC_SEL_cost(nmodes, nmodals, nthc, MC, beth = 20, subsel_a = True, diff_angles = True, verbose = False):
  """
  Code to calculate the Toffoli and qubit cost of state preparation in block encoding a vibrational Hamiltonian expressed in terms of THC decomposition.

  Args:
    beth: Number of bits used in approximating the rotation angles (default is 20)
  """

  if type(nthc) is int:
    R_sum = nmodes*nthc
  else:
    R_sum = np.sum(np.array(nthc) - 1)
    if subsel_a == False:
      subsel_a = True
      if verbose:
        print ("Mode dependent THC rank requires Subselect A circuit")

    
  #Cost of orbital rotation
  orb_rot_cost = 2*(nmodals-1)*(beth-2)

  #Cost of subselect:
  if subsel_a == True:
    subsel_cost = 2*(nmodes*nmodals + R_sum - nmodals - 1) + 2*orb_rot_cost
  else:
    subsel_cost = 2*(nmodes-1)*(nmodals+1) + 2*orb_rot_cost + 2*((nthc-1)*((nmodals-1)*beth + 1) - 1)

  #Total Select cost
  C_S = MC*subsel_cost + 2*R_sum + np.ceil(np.log2(MC))
  if (diff_angles == True) and (MC == 3):
    C_S += 4*R_sum

  #Qubit cost
  C_S_qubits = nmodes*nmodals + nmodals*beth - 2


  return C_S, C_S_qubits




#Cost of QPE of THC Hamiltonian
def get_THC_QPE_cost(lmbda, eps, nmodes, nmodals, nthc, MC, br = 20, aleph = 20, beth = 20, subsel_a = True, PREP_vol = None, dirty = False, clean = False, diff_angles = True, verbose = True):
  
  #Number of terms in LCU
  if type(nthc) is int:
    S = int(nmodes*nmodals + (nmodes*(nmodes - 1)/2)*nthc**2)        # Number of |ip> + number of |iu>|jv>
    if MC == 3:
      S += int((nmodes*(nmodes - 1)*(nmodes-2)/6)*nthc**3)           # Number of |iu>|jv>|kw>
    R = nthc
  else:
    R_sum = 0
    for i in range (nmodes):
      for j in range (i):
        R_sum += nthc[i] * nthc[j]
    S = int(nmodes*nmodals + R_sum)
    R = max(nthc)

  if PREP_vol is not None:
    S = PREP_vol

  if dirty == True:
    dirty = nmodes*nmodals + aleph + nmodals*beth - 2                 #The system qubits, aleph qubits from alias sampling, and the ancillas for orbital rotation are idle dirty qubits


  C_P, C_Pdag, C_P_qubits = get_THC_PREP_UNPREP_cost(nmodes, nmodals, S, R, nthc, MC, br, aleph, PREP_vol, dirty, clean, verbose)
  C_S, C_S_qubits = get_THC_SEL_cost(nmodes, nmodals, nthc, MC, beth, subsel_a, diff_angles, verbose)

  if type(nthc) is not int:
    R = max(nthc)
  else:
    R = nthc

  #Number of steps in QPE i.e. number of access to walker operator
  QPE_nsteps = np.ceil(np.pi*lmbda/(2*eps))
  #Cost of single walker = Cost of block encoding + cost of controlled reflection
  C_BE = C_P + C_S + C_Pdag
  C_W = C_BE + np.ceil(np.log2(S)) + aleph + 1


  #Qubit cost of Walker
  C_W_qubits = C_P_qubits + C_S_qubits
  m = 2*MC*(np.ceil(np.log2(nmodes)) + np.ceil(np.log2(R))) + aleph + 2 + 2*(np.ceil(np.log2(MC)))
  QROAM_cost, QROAM_clean, QROAM_dirty = get_QROAM_cost(S, m, dirty, clean, nmodes, nmodals)
  extras = min(QROAM_clean, nmodals*beth - 2)                                              #We will only need the max of these to as ancillas are reused
  C_W_qubits -= extras                                                                        #Subtract the extra ancillas which are not necessary.

  #Total Toffoli cost of THC LCU
  C_THC = QPE_nsteps*C_W

  #Totoal qubit cost of THC LCU
  N_THC = int(2*np.ceil(np.log2(QPE_nsteps + 1)) + C_W_qubits)

  if verbose:
    print (f"\nCost of PREP: {C_P: 0.2e}")
    print (f"Cost of SEL: {C_S: 0.2e}")
    print (f"Cost of BE: {C_BE: 0.2e}")
    print (f"LCU 1-norm: {lmbda: 0.2e} cm-1")
    print (f"# of walks: {QPE_nsteps: 0.2e}")
    print (f"Toffoli cost of THC LCU: {C_THC: 0.5e}")
    print (f"Qubit cost of THC LCU: {N_THC}")
  
  return C_THC, N_THC

