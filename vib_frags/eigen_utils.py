import os
from numba import njit, prange
import numpy as np


# 1. Stop JAX from preallocating all memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 2. (Optional) If you still see issues, force it to use a 'platform' allocator
# This makes JAX behave more like standard Python, allocating/freeing on demand.
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


#To obtain extremal eigenvalues of a vibrational Hamiltonian
import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard as lobpcg
import jax
from functools import partial
import optax
from . import tensor_utils as tu
import psutil
import os
import gc
jax.config.update("jax_enable_x64", True)
from scipy.sparse.linalg import LinearOperator, eigsh




def print_memory_usage(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # RSS (Resident Set Size) is the non-swapped physical memory user
    print(f"[{step_name}] Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")


def print_memory_usage2():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # Get system-wide memory stats
    vm = psutil.virtual_memory()
    # RSS (Resident Set Size) is the non-swapped physical memory user
    print(f"Process Usage: {mem_info.rss / 1024 ** 2:.2f} MB | "
          f"System Available: {vm.available / 1024 ** 2:.2f} MB | "
          f"System Used: {vm.percent}%")




@jax.jit
def uni_on_state(U, state):
  """
  Apply a unitary operator on a state vector represented as a tensor

  Parameters
  ----------
  U: jax.ndarray()
    A tensor of shape (nmodes, nmodals, nmodals) representing a unitary operator on each mode
  state: jax.ndarray()
    A tensor of shape (nmodals, nmodals, ... , nmodals) representing a state vector. 
    The number of dimensions of A is nmodes
    
  Returns
  -------
  jax.ndarray()
    A tensor of shape state.shape representing the state vector after applying the unitary operator
  """

  def apply_single_mode(current_state, u_matrix):
    # 1. Contract the unitary (u_matrix) with the FIRST mode (axis 0) of the state.
    #    u_matrix shape: (p, a)
    #    state shape:    (a, b, c, ...)
    #    Result shape:   (p, b, c, ...)
    res = jnp.tensordot(u_matrix, current_state, axes=([1], [0]))
    
    # 2. Move the processed axis (0) to the very end.
    #    This shifts all other modes to the left.
    #    Result shape: (b, c, ..., p)
    #    Now, the "next" mode (b) is at axis 0, ready for the next step in the scan.
    res = jnp.moveaxis(res, 0, -1)
    
    return res, None

    # Scan loops over the first dimension of U (the modes)
  final_state, _ = jax.lax.scan(apply_single_mode, state, U)
  return final_state




@jax.jit
def diag_to_ten(D):
    """
    Obtain tensor representation of a direct sum of diagonal operators
    
    Parameters
    ----------
    D: jax.ndarray
        Shape (nmodes, nmodals).
        
    Returns
    -------
    jax.ndarray
        Tensor of shape (nmodals,)*nmodes representing the direct sum of the diagonal operators.
    """
    nmodes = D.shape[0]
    nmodals = D.shape[1]

    #Think of the tensor representation of the diagonal operator as a potential
    # We will build the "Total Potential Grid" starting from zeros
    initial_potential = jnp.zeros((nmodals,)*nmodes)

    def accumulate_and_rotate(current_potential, d_vec):
        # 1. Reshape d_vec to broadcast against the FIRST dimension (Axis 0)
        #    d_vec shape: (nmodals,)
        #    We need:     (nmodals, 1, 1, ...)
        #    So it adds values to Axis 0 of current_potential
        reshaped_d = d_vec.reshape((d_vec.shape[0],) + (1,) * (nmodes - 1))
        
        # 2. Add the current diagonal to the potential
        #    (This represents the term D_k acting on mode k)
        new_potential = current_potential + reshaped_d
        
        # 3. Cyclic Shift: Move Axis 0 to the end (-1)
        #    Just like the unitary case, this brings the next physical mode 
        #    to the front (Axis 0) for the next iteration.
        new_potential = jnp.moveaxis(new_potential, 0, -1)
        
        return new_potential, None

    # Scan over D. Each step adds one mode's diagonal and rotates.
    # After 'nmodes' steps, we have rotated the tensor fully back to its original alignment.
    total_potential, _ = jax.lax.scan(accumulate_and_rotate, initial_potential, D)
    
    return total_potential




@jax.jit
def diag_on_state(D_ten, state):
  """
  Apply a direct sum of diagonal operator represented as a tensor on a state vector

  Parameters
  ----------
  D_ten: jax.ndarray()
    A tensor of shape (nmodals,)*nmodes
  state: jax.ndarray()
    A tensor of shape (nmodals,)nmodes

  Returns
  -------
  jax.ndarray()
    A tensor of shape state.shape representing the state vector after applying the direct sum of diagonal operators
  """
  return D_ten * state






@jax.jit
def eff_diag_on_state(Dk, state):
    """
    Computes sum of diagonal operators: H|psi> = (D1 + D2 + ... + Dn)|psi>
    Uses Scan to process one mode at a time.
    """
    # D_vecs shape: (nmodes, nmodals)
    
    def scan_body(carry, Dki):
        # unpack carry: (rotated_state, rotated_accumulator)
        curr_state, curr_acc = carry
        
        # 1. Multiply current axis 0 by the diagonal vector Dki
        #    'p..., p -> p...' broadcasts Dki along the first axis of curr_state
        term = jnp.einsum('p..., p -> p...', curr_state, Dki)
        
        # 2. Add to the accumulator
        new_acc = curr_acc + term
        
        # 3. Rotate BOTH the state and accumulator
        #    This moves axis 0 to the end, bringing the next mode to axis 0
        next_state = jnp.moveaxis(curr_state, 0, -1)
        next_acc = jnp.moveaxis(new_acc, 0, -1)
        # jax.debug.callback(print_memory_usage2)
        
        return (next_state, next_acc), None

    # Initial carry: (Original State, Zero Accumulator)
    init_acc = jnp.zeros_like(state)
    
    # Run scan over the diagonal vectors (modes)
    (final_state, final_acc), _ = jax.lax.scan(scan_body, (state, init_acc), Dk)
    
    # After nmodes rotations, final_acc is back in the original alignment
    jax.clear_caches()
    return final_acc





@jax.jit
def obt_on_state(D, U, state):
  """
  Apply one-body operator to a state using the diagonal representation of the one-body operator

  Parameters
  ----------
  D_ten: jax.ndarray()
    A tensor of shape (nmodals,)*nmodes representing the diagonal representation of the one-body operator
  U: jax.ndarray()
    A tensor of shape (nmodes, nmodals, nmodals) representing the unitary diagonalizing the one-body operator
  state: jax.ndarray()
    A tensor of shape (nmodals, nmodals, ... , nmodals) representing a state vector. 
    The number of dimensions of A is nmodes

  Returns
  -------
  jax.ndarray()
    A tensor of shape state.shape representing the state vector after applying the one-body operator
  """
  Ustate = uni_on_state(jnp.transpose(U, (0, 2, 1)), state)
  # D_ten = diag_to_ten(D)
  # DUstate = D_ten * Ustate
  DUstate = eff_diag_on_state(D, Ustate)
  obtstate = uni_on_state(U, DUstate) 
  jax.clear_caches()
  return obtstate




@jax.jit
def obt2_on_state(D, U, state):
  """
  Apply one-body operator squared to a state using the diagonal representation of the one-body operator

  Parameters
  ----------
  D_ten: jax.ndarray()
    A tensor of shape (nmodals,)*nmodes representing the diagonal representation of the one-body operator
  U: jax.ndarray()
    A tensor of shape (nmodes, nmodals, nmodals) representing the unitary diagonalizing the one-body operator
  state: jax.ndarray()
    A tensor of shape (nmodals, nmodals, ... , nmodals) representing a state vector. 
    The number of dimensions of A is nmodes

  Returns
  -------
  jax.ndarray()
    A tensor of shape state.shape representing the state vector after applying the square of the one-body operator
  """
  Ustate = uni_on_state(jnp.transpose(U, (0, 2, 1)), state)
  # D_ten = diag_to_ten(D)
  DUstate = eff_diag_on_state(D, Ustate)
  # jax.debug.callback(print_memory_usage2)
  # DUstate = D_ten * Ustate
  D2Ustate = eff_diag_on_state(D, DUstate)
  # D2Ustate = D_ten * DUstate
  obt2state = uni_on_state(U, D2Ustate) 
  # jax.debug.callback(print_memory_usage2)
  jax.clear_caches()
  return obt2state




@partial(jax.jit, static_argnames=('nmodes', 'nmodals'))
def Ham_on_state(obtD, obtU, tbtD, tbtU, tbt_eigs, flt_state, nmodes, nmodals):
    # 1. Reshape flat state to tensor
    state = flt_state.reshape((nmodals,) * nmodes)
    
    # 2. Apply One-Body Term (Base)
    newstate = obt_on_state(obtD, obtU, state)

    # 3. Define the Scan Body (The Accumulator Loop)
    def scan_body(accumulated_sum, fragment_data):
        # Unpack the current fragment's data
        D_k, U_k, eig_k = fragment_data
        
        # Calculate term: eig_k * (U D^2 U^dagger state)
        op_result = obt2_on_state(D_k, U_k, state)
        contribution = eig_k * op_result
        
        # Update the running sum
        new_sum = accumulated_sum + contribution
        
        # Return: (New Carry, Output to Stack)
        # CRITICAL: We return 'None' as the second argument. 
        # This tells JAX NOT to store the history of every fragment, saving huge memory.
        return new_sum, None


    # 5. Run the scan
    # jax.lax.scan iterates over the 0-th dimension of the tuple (tbtD, tbtU, tbt_eigs)
    total_fragment_sum, _ = jax.lax.scan(scan_body, newstate, (tbtD, tbtU, tbt_eigs))

    # 6. Sum and Flatten
    final_state = total_fragment_sum.ravel()

    return final_state







def get_ground_state(obt, tbt, tbt_cutoff = 1e-3, max_iter = 10000, tol = 1e-3, lr=1e-3):
  """
  Function to obtain the ground state of the Hamiltonian defined by the input one, two, and/or three body tensor.

  Parameters
  ----------
  obt: np.ndarray()
    One-body tensor of shape (nmodes, nmodals, nmodals)
  tbt: np.ndarray()
    Two-body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
  trbt: np.ndarray()
    Three-body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
  tbt_cutoff: float
    Cutoff to ignore small terms in the eigen decomposition of the reshaped tbt tensor

  Returns
  -------
  float
    Ground state energy
  np.ndarray()
    Ground state wavefunction
  """

  nmodes = obt.shape[0]
  nmodals = obt.shape[-1]

  #Obtaining eigenvalues and eigenvectors of OBT
  U0 = np.zeros((nmodes, nmodals, nmodals))
  D0 = np.zeros((nmodes, nmodals))
  for i in range (nmodes): 
    obt_i = obt[i]                                           # The negative sign is used since lobpcg finds largest and not smallest eigenvector
    D0[i], U0[i] = np.linalg.eigh(obt_i)
  # D0_ten = diag_to_ten(D0)
  
  #Performing eigen decomposition of tbt
  G_mat = tbt.reshape((nmodes*nmodals**2, nmodes*nmodals**2))
  G_val, G_vec = np.linalg.eigh(G_mat)
  G_val_mask = np.argsort(np.abs(G_val))[::-1]
  G_val = G_val[G_val_mask]
  G_vec = G_vec[:, G_val_mask]
  
  G_val_mask = np.where(np.abs(G_val) > tbt_cutoff)
  G_val = G_val[G_val_mask]                                  # The negative sign is used since lobpcg finds largest and not smallest eigenvector
  nfrags = len(G_val)
  G_vec_ten = G_vec[:, G_val_mask].reshape((nmodes, nmodals, nmodals, nfrags))  

  Us = np.zeros((nfrags, nmodes, nmodals, nmodals))
  # Ds_ten = np.zeros((nfrags,)+(nmodals,)*nmodes)
  Ds = np.zeros((nfrags, nmodes, nmodals))
  for f in range (nfrags):
    Dsf = np.zeros((nmodes, nmodals))
    for i in range (nmodes):
      obt_i = G_vec_ten[i, :, :, f]
      Dsf[i, :], Us[f, i, :, :] = np.linalg.eigh(obt_i)
    Ds[f] = Dsf
    # Ds_ten[f] = np.array(diag_to_ten(Dsf))

  print ("Diagonalizing two-body fragments complete. Procedeing to ground energy estimation.")
  print ("Number of tbt frags = ", nfrags)

  # Usage inside your ground_state function:
  print_memory_usage("Before JAX Arrays")
  D0, U0, Ds, Us, G_val = jnp.array(D0), jnp.array(U0), jnp.array(Ds), jnp.array(Us), jnp.array(G_val)
  print_memory_usage("After JAX Arrays")

  # obt_mat = tu.obt2_proj_mat(obt)
  # tbt_mat = tu.tbt2_proj_mat(tbt)
  # H_mat = obt_mat + tbt_mat
  # H_mat = jnp.array(H_mat.toarray())

  @jax.jit
  def matvec(flt_state):
    return Ham_on_state(D0, U0, Ds, Us, G_val, flt_state, nmodes, nmodals)
    # return (H_mat @ flt_state.reshape((nmodals**nmodes,1))).ravel()

  @jax.jit
  def matvec_batched(block_state):
    return jax.vmap(matvec, in_axes=1, out_axes=1)(block_state)

  init_state = np.random.random((nmodals**nmodes))
  init_state = np.zeros(nmodals**nmodes)
  init_state[0] = 1
  # init_state = init_state/jnp.linalg.norm(init_state)
  init_state = jnp.array(init_state.reshape((nmodals**nmodes,1)))

  print_memory_usage("Before lobpcg")
  vals, vecs, history = lobpcg(matvec_batched, init_state, m=max_iter, tol=tol)
  print_memory_usage("After lobpcg")

  jax.clear_caches()
  print ("Number of lobpcg iterations: ", history)
  return vals, vecs

  obt_mat = tu.obt2_proj_mat(obt)
  tbt_mat = tu.tbt2_proj_mat(tbt)
  H_mat = obt_mat + tbt_mat
  H_mat = jnp.array(H_mat.toarray())



  @jax.jit
  def energy_loss(psi):
      # 1. Apply H to psi (your custom logic)
      # H_psi = matvec(psi) 
      psi_vec = psi.reshape((nmodals**nmodes, 1))
      H_psi = H_mat @ psi_vec
      
      # 2. Compute expectation value <psi|H|psi>
      # Assuming real inputs for simplicity; use jnp.vdot for complex
      numerator = jnp.vdot(psi, H_psi)
      denominator = jnp.vdot(psi, psi)
      
      return (numerator / denominator).real

  # Optimize
  optimizer = optax.adam(learning_rate=lr)
  psi = np.random.random((nmodals**nmodes,1))
  psi = np.zeros(nmodals**nmodes,)
  psi[0] = 1
  opt_state = optimizer.init(psi)

  @jax.jit
  def update_step(psi, opt_state):
      loss, grads = jax.value_and_grad(energy_loss)(psi)
      updates, opt_state = optimizer.update(grads, opt_state, psi)
      psi = optax.apply_updates(psi, updates)
      return psi, opt_state, loss

  for i in range(max_iter):
      psi, opt_state, loss = update_step(psi, opt_state)
      if i%50 == 0:
        print(f"Ground State Energy: {loss}")
  
  return loss, psi












def get_ground_state_scipy(obt, tbt, tbt_cutoff = 1e-3, max_iter = 10000, tol = 1e-3, lr=1e-3, ncv=5):
  """
  Function to obtain the ground state of the Hamiltonian defined by the input one, two, and/or three body tensor.

  Parameters
  ----------
  obt: np.ndarray()
    One-body tensor of shape (nmodes, nmodals, nmodals)
  tbt: np.ndarray()
    Two-body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
  trbt: np.ndarray()
    Three-body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
  tbt_cutoff: float
    Cutoff to ignore small terms in the eigen decomposition of the reshaped tbt tensor

  Returns
  -------
  float
    Ground state energy
  np.ndarray()
    Ground state wavefunction
  """

  nmodes = obt.shape[0]
  nmodals = obt.shape[-1]

  #Obtaining eigenvalues and eigenvectors of OBT
  U0 = np.zeros((nmodes, nmodals, nmodals))
  D0 = np.zeros((nmodes, nmodals))
  for i in range (nmodes): 
    obt_i = obt[i]                                           # The negative sign is used since lobpcg finds largest and not smallest eigenvector
    D0[i], U0[i] = np.linalg.eigh(obt_i)
  # D0_ten = diag_to_ten(D0)
  
  #Performing eigen decomposition of tbt
  G_mat = tbt.reshape((nmodes*nmodals**2, nmodes*nmodals**2))
  G_val, G_vec = np.linalg.eigh(G_mat)
  G_val_mask = np.argsort(np.abs(G_val))[::-1]
  G_val = G_val[G_val_mask]
  G_vec = G_vec[:, G_val_mask]
  
  G_val_mask = np.where(np.abs(G_val) > tbt_cutoff)
  G_val = G_val[G_val_mask]                                  # The negative sign is used since lobpcg finds largest and not smallest eigenvector
  nfrags = len(G_val)
  G_vec_ten = G_vec[:, G_val_mask].reshape((nmodes, nmodals, nmodals, nfrags))  

  Us = np.zeros((nfrags, nmodes, nmodals, nmodals))
  # Ds_ten = np.zeros((nfrags,)+(nmodals,)*nmodes)
  Ds = np.zeros((nfrags, nmodes, nmodals))
  for f in range (nfrags):
    Dsf = np.zeros((nmodes, nmodals))
    for i in range (nmodes):
      obt_i = G_vec_ten[i, :, :, f]
      Dsf[i, :], Us[f, i, :, :] = np.linalg.eigh(obt_i)
    Ds[f] = Dsf
    # Ds_ten[f] = np.array(diag_to_ten(Dsf))

  print ("Diagonalizing two-body fragments complete. Procedeing to ground energy estimation.")
  print ("Number of tbt frags = ", nfrags)

  # Usage inside your ground_state function:
  print_memory_usage("Before JAX Arrays")
  D0, U0, Ds, Us, G_val = jnp.array(D0), jnp.array(U0), jnp.array(Ds), jnp.array(Us), jnp.array(G_val)
  print_memory_usage("After JAX Arrays")

  # obt_mat = tu.obt2_proj_mat(obt)
  # tbt_mat = tu.tbt2_proj_mat(tbt)
  # H_mat = obt_mat + tbt_mat
  # H_mat = jnp.array(H_mat.toarray())


  def matvec_wrapper(flt_state):
    return Ham_on_state(D0, U0, Ds, Us, G_val, flt_state, nmodes, nmodals)
    # return (H_mat @ flt_state.reshape((nmodals**nmodes,1))).ravel()

  dim = nmodals**nmodes


  A_op = LinearOperator((dim, dim), matvec=matvec_wrapper, dtype=np.float64)

  # 3. Solve for the Ground State
  print("Starting iterative solver...")
  print_memory_usage("Before lobpcg")
  eigenvalues, eigenvectors = eigsh(A_op, k=1, which='SA', maxiter=max_iter, tol=tol, ncv=ncv)
  print_memory_usage("After lobpcg")

  # print(f"Ground State Energy: {eigenvalues[0]:.6f}")

  jax.clear_caches()

  del U0, D0, Us, Ds, G_val, G_mat, G_vec, G_vec_ten, A_op
  gc.collect()


  return eigenvalues[0], eigenvectors











@njit(parallel=True, fastmath=True)
def _apply_unitary_kernel_numba(state_in, state_out, U, pre_dim, m_dim, post_dim, adjoint=False):
    """
    Core Numba kernel. 
    """
    for i in prange(pre_dim):
        for j in range(m_dim):
            # 1. Zero out the output row first
            for k in range(post_dim):
                state_out[i, j, k] = 0.0
            
            # 2. Accumulate Matrix Product
            for p in range(m_dim):
                # Correct Adjoint logic (even for Real matrices, it's safer)
                if adjoint:
                    u_val = U[p, j]
                else:
                    u_val = U[j, p]
                
                # Optimization for sparse U
                if abs(u_val) > 1e-15: 
                    for k in range(post_dim):
                        state_out[i, j, k] += u_val * state_in[i, p, k]

@njit(fastmath=True)
def uni_on_state_numba(U, state, adjoint=False):
    """
    Applies unitaries to state.
    Safe against overwriting input and non-contiguous memory.
    """
    # SAFETY 1: Create a COPY of the input state.
    # Your previous code modified 'state' in-place every 2nd iteration.
    current_state = state.astype(np.float64).copy()
    
    # SAFETY 2: Ensure contiguous memory so 'reshape' always returns a VIEW, not a copy.
    current_state = np.ascontiguousarray(current_state)
    temp_buffer = np.empty_like(current_state)
    
    nmodes = U.shape[0]
    nmodals = U.shape[1]
    
    for mode in range(nmodes):
        pre_dim = nmodals ** mode
        post_dim = nmodals ** (nmodes - 1 - mode)
        
        # Reshape creates views into the contiguous arrays
        in_view = current_state.reshape((pre_dim, nmodals, post_dim))
        out_view = temp_buffer.reshape((pre_dim, nmodals, post_dim))
        
        _apply_unitary_kernel_numba(in_view, out_view, U[mode], pre_dim, nmodals, post_dim, adjoint)
        
        # Swap buffers
        current_state, temp_buffer = temp_buffer, current_state

    return current_state







@njit(parallel=True, fastmath=True)
def _add_mode_potential(potential_buffer, d_vec, pre_dim, m_dim, post_dim):
    """
    Adds a diagonal vector (d_vec) to the potential_buffer, broadcasting it
    across the 'pre' and 'post' dimensions.
    
    potential_buffer shape: (pre_dim, m_dim, post_dim)
    d_vec shape: (m_dim,)
    """
    for i in prange(pre_dim):
        for j in range(m_dim):
            # Cache the value to add for this slice
            val = d_vec[j]
            # Inner loop is contiguous and vectorizes perfectly
            for k in range(post_dim):
                potential_buffer[i, j, k] += val




@njit(fastmath=True)
def _multiply_potential_and_state(output_buffer, state_tensor, square_potential=False):
    flat_out = output_buffer.ravel()
    flat_state = state_tensor.ravel()
    n = flat_out.size
    
    # Numba will compile two versions of this loop efficiently
    if square_potential:
        for i in prange(n):
            # V^2 * psi
            val = flat_out[i]
            flat_out[i] = (val * val) * flat_state[i]
    else:
        for i in prange(n):
            # V * psi
            flat_out[i] *= flat_state[i]





@njit(fastmath=True)
def eff_diag_on_state_numba(D, state, square_potential=False):
    """
    Efficiently computes (D1 + D2 + ... + Dn) |psi> on CPU.
    
    Parameters
    ----------
    D : np.ndarray
        Shape (nmodes, nmodals). The diagonal entries for each mode.
    state : np.ndarray
        Shape (nmodals, nmodals, ...). The state tensor.
        
    Returns
    -------
    np.ndarray
        The result of the operation.
    """
    # 1. Allocate Output Buffer (Initialized to 0.0)
    #    We will first build the 'Total Potential' sum inside this buffer.
    #    Use result_type to ensure we handle complex states correctly.
    output = np.zeros_like(state)
    
    nmodes = D.shape[0]
    nmodals = D.shape[1]
    
    # 2. Accumulate Diagonal Contributions
    #    For each mode, we add its diagonal vector to the output tensor,
    #    broadcasting over all other dimensions.
    for mode in range(nmodes):
        # Calculate strides ("virtual reshape")
        pre_dim = nmodals ** mode
        post_dim = nmodals ** (nmodes - 1 - mode)
        
        # Create a view of the output for this specific mode's alignment
        out_view = output.reshape((pre_dim, nmodals, post_dim))
        
        # Add this mode's diagonal to the total sum
        _add_mode_potential(out_view, D[mode], pre_dim, nmodals, post_dim)
        
    # 3. Apply to State
    #    Now 'output' contains the sum of all diagonals (the potential V).
    #    We simply multiply: Result = V * State
    _multiply_potential_and_state(output, state, square_potential)
    
    return output






@njit(fastmath=True)
def obt_on_state_numba(D, U, state):
  """
  Apply one-body operator to a state using the diagonal representation of the one-body operator

  Parameters
  ----------
  D_ten: jax.ndarray()
    A tensor of shape (nmodals,)*nmodes representing the diagonal representation of the one-body operator
  U: jax.ndarray()
    A tensor of shape (nmodes, nmodals, nmodals) representing the unitary diagonalizing the one-body operator
  state: jax.ndarray()
    A tensor of shape (nmodals, nmodals, ... , nmodals) representing a state vector. 
    The number of dimensions of A is nmodes

  Returns
  -------
  jax.ndarray()
    A tensor of shape state.shape representing the state vector after applying the one-body operator
  """
  Ustate = uni_on_state_numba(U, state, adjoint=True)
  DUstate = eff_diag_on_state_numba(D, Ustate)
  obtstate = uni_on_state_numba(U, DUstate) 
  return obtstate




@njit(fastmath=True)
def obt2_on_state_numba(D, U, state):
  """
  Apply one-body operator squared to a state using the diagonal representation of the one-body operator

  Parameters
  ----------
  D_ten: jax.ndarray()
    A tensor of shape (nmodals,)*nmodes representing the diagonal representation of the one-body operator
  U: jax.ndarray()
    A tensor of shape (nmodes, nmodals, nmodals) representing the unitary diagonalizing the one-body operator
  state: jax.ndarray()
    A tensor of shape (nmodals, nmodals, ... , nmodals) representing a state vector. 
    The number of dimensions of A is nmodes

  Returns
  -------
  jax.ndarray()
    A tensor of shape state.shape representing the state vector after applying the square of the one-body operator
  """
  Ustate = uni_on_state_numba(U, state, adjoint=True)
  D2Ustate = eff_diag_on_state_numba(D, Ustate, square_potential=True)
  obt2state = uni_on_state_numba(U, D2Ustate) 
  return obt2state






@njit(fastmath=True)
def Ham_on_state_numba(obtD, obtU, tbtD, tbtU, tbt_eigs, state, nmodes, nmodals):
    """
    Numba-optimized Hamiltonian action on state.
    
    Parameters
    ----------
    obtD, obtU : One-Body Term diagonals and unitaries
    tbtD, tbtU : Two-Body Term diagonals and unitaries
    tbt_eigs   : Weights (eigenvalues) for the TBT terms
    flt_state  : Flattened input state vector
    nmodes, nmodals : System dimensions
    """
    
    # 1. Reshape and ensure Complex Type
    #    We force complex128 to ensure the helper functions (which use empty_like)
    #    allocate buffers capable of holding complex unitary results.
    #    (Constructing the shape tuple works in Numba if nmodes is constant or simple integer)
    
    # Create the shape tuple (nmodals, nmodals, ..., nmodals)
    # Note: In some Numba versions, dynamic tuple construction can be strict.
    # If this reshape line causes issues, pass 'state' already reshaped/casted.
    # We use a manual reshape calculation or assume flat access for helpers if they supported it,
    # but your helpers expect tensor shape. 
    # NOTE: Assuming nmodes is known/small enough for this tuple construction.
    # shape = (nmodals,) * nmodes
    # state = flt_state.reshape(shape)

    # 2. Apply One-Body Term (OBT)
    #    This initializes our accumulator.
    #    Allocation: 1x State Size (internal to obt_on_state_numba)
    if (obtD is not None) and (obtU is not None):
      accumulator = obt_on_state_numba(obtD, obtU, state)
    else:
      accumulator = np.zeros_like(state)


    # 3. Apply Two-Body Terms (TBT)
    n_frags = tbt_eigs.shape[0]

    # We flatten the accumulator view once to make the inner addition loop trivial
    acc_flat = accumulator.ravel()
    
    for k in range(n_frags):
        # Apply the operator part: U * D^2 * U^dag * state
        # Allocation: 1x State Size (Returned by obt2_on_state_numba)
        # Note: obt2_on_state_numba must call 'uni_on_state_numba' internally.
        term_tensor = obt2_on_state_numba(tbtD[k], tbtU[k], state)
        
        weight = tbt_eigs[k]
        
        # Optimization: Manual In-Place Accumulation
        # Instead of doing `accumulator += weight * term_tensor` (which creates a temp array),
        # we loop manually. Numba fuses this perfectly.
        term_flat = term_tensor.ravel()
        
        for i in prange(acc_flat.size):
            acc_flat[i] += weight * term_flat[i]

    return accumulator.ravel()





def get_eigen_state_numba(obt, tbt, tbt_cutoff = 1e-5, nfrags = None, which='SA', k=1, max_iter = 1000, tol = 1e-5, ncv=None):
  """
  Function to obtain an eigenstate of the Hamiltonian defined by the input one, two, and/or three body tensor.

  Parameters
  ----------
  obt: np.ndarray()
    One-body tensor of shape (nmodes, nmodals, nmodals)
  tbt: np.ndarray()
    Symmetrized two-body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
  trbt: np.ndarray()
    Three-body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
  tbt_cutoff: float
    Cutoff to ignore small terms in the eigen decomposition of the reshaped tbt tensor

  Returns
  -------
  float
    Ground state energy
  np.ndarray()
    Ground state wavefunction
  """

  nmodes = tbt.shape[0]
  nmodals = tbt.shape[-1]

  #Obtaining eigenvalues and eigenvectors of OBT
  if obt is not None:
    U0 = np.zeros((nmodes, nmodals, nmodals))
    D0 = np.zeros((nmodes, nmodals))
    for i in range (nmodes): 
      obt_i = obt[i]                                           
      D0[i], U0[i] = np.linalg.eigh(obt_i)
  else:
    D0, U0 = None, None
  # D0_ten = diag_to_ten(D0)
  
  #Performing eigen decomposition of tbt
  G_mat = tbt.reshape((nmodes*nmodals**2, nmodes*nmodals**2))
  G_val, G_vec = np.linalg.eigh(G_mat)
  G_val_mask = np.argsort(np.abs(G_val))[::-1]
  G_val = G_val[G_val_mask]
  G_vec = G_vec[:, G_val_mask]
  
  if nfrags is None:
    G_val_mask = np.where(np.abs(G_val) > tbt_cutoff)
    G_val = G_val[G_val_mask]                        
    nfrags = len(G_val)          
    G_vec_ten = G_vec[:, G_val_mask].reshape((nmodes, nmodals, nmodals, nfrags))
  elif type(nfrags) is int:
    G_val_mask = G_val_mask[:nfrags]
    G_val = G_val[G_val_mask]                        
    nfrags = len(G_val)          # The negative sign is used since lobpcg finds largest and not smallest eigenvector
    G_vec_ten = G_vec[:, G_val_mask].reshape((nmodes, nmodals, nmodals, nfrags))
  elif (type(nfrags) is tuple) or (type(nfrags) is list):
    npfrags, nnfrags = nfrags
    pos_mask = np.where(G_val > 0)[0]
    neg_mask = np.where(G_val < 0)[0]
    png_mask = np.concatenate((pos_mask[:npfrags], neg_mask[:nnfrags]))
    nfrags = npfrags + nnfrags
    G_val = G_val[png_mask]
    G_vec_ten = G_vec[:, png_mask].reshape((nmodes, nmodals, nmodals, nfrags))  


  Us = np.zeros((nfrags, nmodes, nmodals, nmodals))
  # Ds_ten = np.zeros((nfrags,)+(nmodals,)*nmodes)
  Ds = np.zeros((nfrags, nmodes, nmodals))
  for f in range (nfrags):
    Dsf = np.zeros((nmodes, nmodals))
    for i in range (nmodes):
      obt_i = G_vec_ten[i, :, :, f]
      Dsf[i, :], Us[f, i, :, :] = np.linalg.eigh(obt_i)
    Ds[f] = Dsf
    # Ds_ten[f] = np.array(diag_to_ten(Dsf))

  print ("Diagonalizing two-body fragments complete. Procedeing to ground energy estimation.")
  print ("Number of tbt frags = ", nfrags)

  def matvec_wrapper(flt_state):
    state = flt_state.reshape((nmodals,) * nmodes)
    return Ham_on_state_numba(D0, U0, Ds, Us, G_val, state, nmodes, nmodals)

  dim = nmodals**nmodes


  A_op = LinearOperator((dim, dim), matvec=matvec_wrapper, dtype=np.float64)

  # 3. Solve for the Ground State
  print("Starting iterative solver...")
  print_memory_usage("Before eigsh")
  eigenvalues, eigenvectors = eigsh(A_op, k=k, which=which, maxiter=max_iter, tol=tol, ncv=ncv)
  print_memory_usage("After eigsh")

  # print(f"Ground State Energy: {eigenvalues[0]:.6f}")

  del U0, D0, Us, Ds, G_val, G_mat, G_vec, G_vec_ten, A_op
  gc.collect()

  if k == 1:
    return eigenvalues[0], eigenvectors
  else:
    return eigenvalues, eigenvectors