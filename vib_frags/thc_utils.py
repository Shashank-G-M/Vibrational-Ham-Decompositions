from . import tensor_utils as tu
from . import mode2_gfro_utils as g2u
import jax.numpy as jnp
import jax
import optax
from copy import copy

import numpy as np
import scipy as sp
from opt_einsum import contract
from scipy.optimize import minimize


def get_thc_2mode_terms(Tbt, M=1, x0 = None, tol = 1e-1, niter = 100, solver='L-BFGS-B', return_tens = False):
  """
  Obtain components necessary to construct tensor hypercontraction fragments of 2 mode tensor. 
  Error will be raised if the input tbt does not have 8 fold symmetry.

  Parameters
  ----------
  tbt : jnp.ndarray
      A two body tensor of shape (i, p, q, j, r, s).
  M : int, optional
      M decides the factor by which the size of the thc basis differs from the original basis. 
      Default is 6 which measn size of thc basis = 6*nmodes*nmodals.
  x0 : jnp.ndarray(M*nmodes*nmodals + M*nmodes*nmodals*(M*nmodes*nmodals+1)/2), optional
      Initial guess for the parameters of the thc fragment. If None, a random guess is used. Default is None.
  tol : float
      If the L2 norm^2 of the residual tensor is less than tol, the optimization is terminated.
  niter : int
      The maximum number of iterations for the optimization. Default is 1000.
  solver : str
      The solver to the scipy.optimize.minimize function. Default is 'L-BFGS-B'.
  return_tens : bool
      If True, the function also returns the thc fragment tensors. Default is False.

  Returns
  -------
    jnp.ndarray
      The non-orthogonal thc rotation tensor, xi
    jnp.ndarray
      The diagonal thc tensor, zeta
  """
  
  tbt = copy(Tbt)
  nmodes = tbt.shape[0]
  nmodals = tbt.shape[-1]


  def loss_fn(x, tbt):
    xi_mat, zeta_mat = params2_thc_mats(x, M, nmodes, nmodals)
    thc_ten = jnp.einsum('xip, xiq, xy, yjr ,yjs -> ipqjrs', xi_mat, xi_mat, zeta_mat, xi_mat, xi_mat)
    diff = jnp.sum((thc_ten - tbt)**2)
    return diff
  
  # solver = optax.lbfgs()
  # params = jnp.zeros((M*nmodes*nmodals + M*nmodes*nmodals*(M*nmodes*nmodals+1)/2))

  # opt_state = solver.init(params)
  # value_and_grad = optax.value_and_grad_from_state(loss_fn)
  # for _ in range(100):
  #   value, grad = value_and_grad(params, state=opt_state)
  #   updates, opt_state = solver.update(
  #     grad, opt_state, params, value=value, grad=grad, value_fn=f
  #   )
  #   params = optax.apply_updates(params, updates)
  #   print('Objective function: {:.2E}'.format(loss_fn(params, tbt)))



  # Optimizer
  # optimizer = optax.adagrad(learning_rate=0.1)
  optimizer = optax.lbfgs()
  # x = jnp.zeros((int(M*(nmodes*nmodals)**2 + M*nmodes*nmodals*(M*nmodes*nmodals+1)/2)))
  key = jax.random.PRNGKey(0)
  length = int(M*(nmodes*nmodals)**2 + M*nmodes*nmodals*(M*nmodes*nmodals+1)/2)
  x = jax.random.uniform(key, shape=(length,), minval=-1, maxval=1)
  opt_state = optimizer.init(x)
  print ('Optimizer initiated.')

  # @jax.jit
  def step(x, opt_state, tbt):
      value, grad = jax.value_and_grad(loss_fn)(x, tbt)
      # updates, opt_state = optimizer.update(grad, opt_state)
      updates, opt_state = optimizer.update(
                grad,
                opt_state,
                x,  # current parameters
                value=value,  # scalar loss at x
                grad=grad,    # gradient at x
                value_fn=lambda x: loss_fn(x, tbt)  # loss function for line search
            )

      x = optax.apply_updates(x, updates)
      return x, opt_state, value

  # Run optimization
  for i in range(niter):
      x, opt_state, value = step(x, opt_state, tbt)
      print ("Current cost = ", loss_fn(x, tbt))
  
  xi_mat, zeta_mat = params2_thc_mats(x, M, nmodes, nmodals)
  return xi_mat, zeta_mat


  


















def params2_thc_mats(params, M, nmodes, nmodals):
    """
    Convert unique parameters defining THC fragments to their corresponding matrices.

    Parameters
    ----------
    params : array_like
      1D array of parameters of the THC fragments. Should be of length M*(nmodes*nmodals)^2 + M*nmodes*nmodals*(M*nmodes*nmodals + 1)/2
    M : int, optional
      M decides the factor by which the size of the thc basis differs from the original basis. 
    nmodes : int
      The number of modes.
    nmodals : int
      The number of modals.

    Returns
    -------
    jnp.ndarray
      The non-orthogonal thc rotation tensor, xi
    jnp.ndarray
      The diagonal thc tensor, zeta
    """

    xi = params[ : M*(nmodes*nmodals)**2]
    xi_mat = jnp.reshape(xi, (M*nmodes*nmodals, nmodes, nmodals))

    zeta = params[M*(nmodes*nmodals)**2 : ]
    zeta_mat = jnp.zeros((M*(nmodes*nmodals), M*(nmodes*nmodals)))
    idx = 0
    for i in range (M*(nmodes*nmodals)):
      for j in range (i+1):
        zeta_mat = zeta_mat.at[i, j].set(zeta[idx])
        zeta_mat = zeta_mat.at[j, i].set(zeta[idx])
        idx += 1

    return xi_mat, zeta_mat


















def get_thc_2mode_terms_scipy(Tbt, M=1, x0 = None, tol = 1e-1, maxiter = 10000, solver='L-BFGS-B', return_tens = False):
  """
  This use scipy unlike jax in the above functions. Obtain components necessary to construct tensor hypercontraction fragments of 2 mode tensor. 
  Error will be raised if the input tbt does not have 8 fold symmetry.

  Parameters
  ----------
  tbt : np.ndarray
      A two body tensor of shape (i, p, q, j, r, s).
  M : int, optional
      M decides the factor by which the size of the thc basis differs from the original basis. 
      Default is 6 which measn size of thc basis = 6*nmodes*nmodals.
  x0 : np.ndarray(M*nmodes*nmodals + M*nmodes*nmodals*(M*nmodes*nmodals+1)/2), optional
      Initial guess for the parameters of the thc fragment. If None, a random guess is used. Default is None.
  tol : float
      If the L2 norm^2 of the residual tensor is less than tol, the optimization is terminated.
  maxiter : int
      The maximum number of iterations for the optimization. Default is 1000.
  solver : str
      The solver to the scipy.optimize.minimize function. Default is 'L-BFGS-B'.
  return_tens : bool
      If True, the function also returns the thc fragment tensors. Default is False.

  Returns
  -------
    np.ndarray
      The non-orthogonal thc rotation tensor, xi
    np.ndarray
      The diagonal thc tensor, zeta
  """
  
  tbt = copy(Tbt)
  nmodes = tbt.shape[0]
  nmodals = tbt.shape[-1]

  
  
  length = int(M*(nmodes*nmodals)**2 + M*nmodes*nmodals*(M*nmodes*nmodals+1)/2)
  if x0 == None:
    # x0 = np.zeros(length)
    x0 = np.random.uniform(-1, 1, length)

  global x_opt
  x_opt = copy(x0)
  global global_diff
  global_diff = 1000

  def loss_fn(x, tbt):
    xi_mat, zeta_mat = params2_thc_mats_scipy(x, M, nmodes, nmodals)
    thc_ten = contract('xip, xiq, xy, yjr ,yjs -> ipqjrs', xi_mat, xi_mat, zeta_mat, xi_mat, xi_mat)
    diff = np.sum((thc_ten - tbt)**2)
    print ("Cost = ", diff)
    global x_opt, global_diff
    if diff < global_diff:
       global_diff = copy(diff)
       x_opt = copy(x)
    return diff

  result = minimize(loss_fn, x0, tbt, method = solver, tol=tol, options={'maxiter': maxiter})
  # x_opt = result.x

  xi_mat, zeta_mat = params2_thc_mats_scipy(x_opt, M, nmodes, nmodals)
  return xi_mat, zeta_mat


  





def params2_thc_mats_scipy(params, M, nmodes, nmodals):
    """
    This uses scipy and not JAX. Convert unique parameters defining THC fragments to their corresponding matrices.

    Parameters
    ----------
    params : array_like
      1D array of parameters of the THC fragments. Should be of length M*(nmodes*nmodals)^2 + M*nmodes*nmodals*(M*nmodes*nmodals + 1)/2
    M : int, optional
      M decides the factor by which the size of the thc basis differs from the original basis. 
    nmodes : int
      The number of modes.
    nmodals : int
      The number of modals.

    Returns
    -------
    np.ndarray
      The non-orthogonal thc rotation tensor, xi
    np.ndarray
      The diagonal thc tensor, zeta
    """

    xi = params[ : int(M*(nmodes*nmodals)**2)]
    xi_mat = np.reshape(xi, (int(M*nmodes*nmodals), nmodes, nmodals))
    # xi_mat = xi_mat/np.linalg.norm(xi_mat, axis=-1, keepdims=True)

    zeta = params[int(M*(nmodes*nmodals)**2) : ]
    zeta_mat = np.zeros((int(M*(nmodes*nmodals)), int(M*(nmodes*nmodals))))
    idx = 0
    for i in range (int(M*(nmodes*nmodals))):
      for j in range (i+1):
        zeta_mat[i, j] = zeta[idx]
        zeta_mat[j, i] = zeta[idx]
        idx += 1

    return xi_mat, zeta_mat



















def GFRO_2_GGFRO_params(Tbt, frag_params, method = 'bfgs', tol = 1e-8, maxiter = 10000, resume = False):
    """
    The function finds generalized GFRO fragments with lower one-norm starting from GFRO fragments. 
    Here, generalized means the orbital rotation matrix for each fragment will be made non-orthogonal in exchange for lower one-norm.

    Parameters
    ----------
    Tbt : np.array(nmodes, nmodes, nmodals, nmodals, nmodals, nmodals)
      The two-body tensor of the Hamiltonian
    frag_params : list(list(...))
      List of lists of parameters defining GFRO fragments
    
    Returns
    -------
    list(list(...))
      List of lists of parameters defining generalized GFRO fragments
    """

    tbt = copy(Tbt)
    nmodes = tbt.shape[0]
    nmodals = tbt.shape[-1]
    c, u, p = g2u.num_params(nmodals, nmodes) 

    def loss_fn(x, org_frag):
       cart_params = x[:c]
       cart = g2u.construct_cartan_tensor(cart_params, nmodals, nmodes)

       O_params = x[c:]
       NO = np.reshape(O_params, (nmodes, nmodals, nmodals))
       NO = NO/np.linalg.norm(NO, axis=-1, keepdims=True)

       new_frag = contract('lmpqpq,lpa,mqb,lpc,mqd->lmabcd', cart, NO, NO, NO, NO)
       diff = np.sum(np.abs(org_frag - new_frag))**2
       onenorm = np.sum(np.abs(cart))
       loss = diff + onenorm
       print ("Diff = ", diff)
       print ("One norm = ", onenorm)
       return loss

    new_params = []

    for i in range (len(frag_params)):
      param_i = frag_params[i]       
      lam    = param_i[ : c]
      theta  = param_i[c : ]
      
      cart = g2u.construct_cartan_tensor(lam, nmodals, nmodes)
      if resume == True:
        O = np.reshape(theta, (nmodes, nmodals, nmodals))
        O = O/np.linalg.norm(O, axis=-1, keepdims=True)
      else:
        O   = g2u.construct_orthogonal(theta, nmodals, nmodes)
      x0 = list(lam) + list(np.ravel(O))

      GFRO_frag = contract('lmpqpq,lpa,mqb,lpc,mqd->lmabcd', cart, O, O, O, O)

      result = minimize(loss_fn, x0, GFRO_frag, method = method, tol = tol, options = {'maxiter' : maxiter})
      xi_opt = result.x

      new_params.append(list(xi_opt))

      print (f"Final loss value for frag {i}:")
      loss_fn(xi_opt, GFRO_frag)
  
    return new_params











def get_gthc_2mode_terms(Tbt, num_frags = 25, frag_tol = 1e-1, method = 'bfgs', maxiter = 100000, return_tens = False):
    """
    The function implements a greedy version of THC.

    Parameters
    ----------
    Tbt : np.array(nmodes, nmodes, nmodals, nmodals, nmodals, nmodals)
      The two-body tensor of the Hamiltonian
       
    Returns
    -------
    list(list(...))
      List of lists of parameters defining generalized GFRO fragments
    """

    tbt = copy(Tbt)
    nmodes = tbt.shape[0]
    nmodals = tbt.shape[-1]
    c, u, p = g2u.num_params(nmodals, nmodes) 

    def loss_fn(x, org_frag):
       cart, NO = params2_gthc_mats(x, c, nmodes, nmodals)
       frag = contract('lmpqpq,lpa,mqb,lpc,mqd->lmabcd', cart, NO, NO, NO, NO)

       diff = np.sum((org_frag - frag)**2)
       onenorm = np.sum(np.abs(cart))
       loss = diff
       print ("Diff = ", diff)
       print ("One norm = ", onenorm)
       return loss

    new_params = []
    all_frags = []

    for i in range (num_frags):
      x0a = np.random.uniform(-100, 100, (c,)) 
      x0b = np.random.uniform(-1, 1, (nmodes*nmodals**2,)) 
      x0 = np.concatenate((x0a, x0b))
      x0 = list(x0)

      result = minimize(loss_fn, x0, tbt, method = method, tol = 1e-5, options = {'maxiter' : maxiter})
      xi_opt = result.x

      new_params.append(list(xi_opt))

      cart, NO = params2_gthc_mats(xi_opt, c, nmodes, nmodals)
      frag = contract('lmpqpq,lpa,mqb,lpc,mqd->lmabcd', cart, NO, NO, NO, NO)
      tbt = tbt - frag
      final_loss = np.sum(tbt**2)

      print (f"Final loss value for frag {i}: ", final_loss)
      if return_tens == True:
         all_frags.append(frag)

      if final_loss < frag_tol:
         break

    if return_tens == True:
       return new_params, all_frags
    else:
       return new_params








def params2_gthc_mats(params, c, nmodes, nmodals):
    """
    Convert a flatteend list of parameters to cartan tensor and the unitary conjugating the cartan.

    Parameters
    ----------
    params : array_like
      1D array of parameters of the THC fragments. Should be of length M*(nmodes*nmodals)^2 + M*nmodes*nmodals*(M*nmodes*nmodals + 1)/2
    c :
      Number of free parameters defining the cartan tensor.
    nmodes : 
      number of modes
    nmodals : 
      number of modals

    Returns
    -------
    np.ndarray
      The cartan tensor
    np.ndarray
      The non-orthogonal orbital rotation tensor
    """

    cart_params = params[:c]
    cart = g2u.construct_cartan_tensor(cart_params, nmodals, nmodes)

    O_params = params[c:]
    NO = np.reshape(O_params, (nmodes, nmodals, nmodals))
    NO = NO/np.linalg.norm(NO, axis=-1, keepdims=True)

    return cart, NO