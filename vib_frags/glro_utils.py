import numpy as np
import scipy as sp
from . import tensor_utils as tu
from opt_einsum import contract
from scipy.optimize import minimize
from copy import copy

def get_GLRO_terms(Trbt, x0 = None, tol = 1e-1, niter = 100, solver='BFGS', return_tens = False):
  """
  Perform greedy low-rank optimization (GLRO) of super summetric three body tensor.
  This method is equivalent to a symmetric CD decomposition. Returns the necessary quantities to recconstruct a three body tensor
  for each GLRO fragment. Error will be raised if the input Trbt does not have 48 fold symmetry.

  Parameters
  ----------
  Trbt : np.ndarray
      A three body tensor of shape (i, p, q, j, r, s, k, t, u).
  x0 : np.ndarray(nmodes*nmodals + nmodes*nmodals*(nmodals-1)/2), optional
      Initial guess for the parameters of the GLRO fragment. If None, a random guess is used. Default is None.
  tol : float
      If the L1 norm of the residual tensor is less than tol, the optimization is terminated.
  niter : int
      The maximum number of iterations for the optimization. Default is 1000.
  solver : str
      The solver to the scipy.optimize.minimize function. Default is 'BFGS'.
  return_tens : bool
      If True, the function returns the GLRO fragment tensors. Default is False.

  Returns
  -------
  list[np.ndarray(nmodes, nmodals)]
      The coefficients of the diagonalized GLRO fragment for each fragment.
  list[np.ndarray(nmodes, nmodals, nmodals)]
      List of orbital rotation matrices for diagonalizing each fragment.
      The first axis of the ndarray denotes the mode index.
      The second and third axis of the ndarray denote the element of an orthogonal matrix performing the orbital rotation.
  list[np.ndarray(nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)] (Optional)
      The GLRO fragment tensors for each fragment
  """
  
  trbt = copy(Trbt)
  nmodes = trbt.shape[0]
  nmodals = trbt.shape[-1]
  coeff_mats = []
  rot_mats = []
  frag_tens = []


  def cost_func(x, trbt):
    frag_ten = params2_GLRO_trbt(x[::-1], nmodes, nmodals)
    return np.sum(np.abs(trbt - frag_ten)**2)

  res_norm = np.sum(np.abs(trbt)**2)
  for ni in range (niter):
    print (f"Starting iteration {ni+1} of {niter}, residual norm: {res_norm}")
    # if x0 is None:
    x0 = np.random.uniform(-900/((5*ni+1)), 900/((5*ni+1)), nmodes*nmodals + int(nmodes*nmodals*(nmodals-1)/2))
    angles = np.random.uniform(-np.pi/2, np.pi/2, int(nmodes*nmodals*(nmodals-1)/2))
    # angles = np.zeros(int(nmodes*nmodals*(nmodals-1)/2))  # Set angles to zero for initial guess
    x0[nmodes*nmodals:] = angles

    # Reverse the order of x0 to pass cartan tensor elements as the first elements to be optimized
    sol = minimize(cost_func, x0[::-1], args=(trbt,), method=solver, tol=1e-6)

    xopt = sol.x[::-1]
    coeff_mat = np.reshape(xopt[:nmodes*nmodals], (nmodes, nmodals))
    rot_mat = params2rot_mat(xopt[nmodes*nmodals:], nmodes, nmodals)
    coeff_mats.append(coeff_mat)
    rot_mats.append(rot_mat)

    frag_ten = contract('il, jm, kn, ipl, iql, jrm, jsm, ktn, kun -> ipqjrsktu',coeff_mat, coeff_mat, coeff_mat,
                      rot_mat, rot_mat, rot_mat, rot_mat, rot_mat, rot_mat)
    if return_tens:
      frag_tens.append(frag_ten)

    res_norm = np.sum(np.abs(trbt - frag_ten)**2)    
    if res_norm < tol:
      break
    else:
      trbt -= frag_ten     
  print (f"Final residual norm: {res_norm}")
  if return_tens:
    return coeff_mats, rot_mats, frag_tens
  else:
    return coeff_mats, rot_mats
  






def params2rot_mat(params, nmodes, nmodals):
    """
    Convert the parameters of the orbital rotation matrix to the actual rotation matrix.
    
    Parameters
    ----------
    params : np.ndarray
        nmodes*nmodals*(nmodals-1)/2 number of parameters of the orbital rotation matrix.
    nmodes : int
        The number of modes.
    nmodals : int
        The number of modals.
    
    Returns
    -------
    np.ndarray(nmodes, nmodals, nmodals)
        The orbital rotation matrix for each mode.
    """
    temp_ten = np.reshape(params, (nmodes, int(nmodals*(nmodals-1)/2)))
    kappa = np.zeros((nmodals, nmodals))
    rot_mat = np.zeros((nmodes, nmodals, nmodals))
    for i in range (nmodes):
      kappa[np.triu_indices(nmodals, k=1)] = temp_ten[i]
      kappa[np.tril_indices(nmodals, k=-1)] = -temp_ten[i]
      rot_mat[i] = sp.linalg.expm(kappa)
    return rot_mat






def params2_GLRO_trbt(params, nmodes, nmodals):
    """
    Convert parameters defining a GLRO fragment to an actual three body tensor.

    Parameters
    ----------
    params : array_like
      1D array of parameters of the GLRO fragment. Should be of length nmodes * nmodals + nmodes * nmodals *(nmodals-1)/2.
    nmodes : int
      The number of modes.
    nmodals : int
      The number of modals.

    Returns
    -------
    np.ndarray
      The three body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals).
    """

    params1 = params[:nmodes*nmodals]
    coeff_mat = np.reshape(params1, (nmodes, nmodals))
    params2 = params[nmodes*nmodals:]
    rot_mat = params2rot_mat(params2, nmodes, nmodals)
    frag_ten = contract('il, jm, kn, ipl, iql, jrm, jsm, ktn, kun -> ipqjrsktu',coeff_mat, coeff_mat, coeff_mat,
                        rot_mat, rot_mat, rot_mat, rot_mat, rot_mat, rot_mat)
    return frag_ten












#Writing Trbt fragments as sum of products of different obt fragments. If the three obts are the same, then it is identical to the above method.
def get_GLROv2_terms(Trbt, x0 = None, tol = 1e-1, niter = 100, solver='BFGS', return_tens = False):
  """
  Perform greedy low-rank optimization (GLRO) of super summetric three body tensor. Express the tensor as a sum of products of different
  low-rank tensors. This method is equivalent to a non-symmetric CD decomposition. Returns the necessary quantities to recconstruct a three body tensor
  for each GLROv2 fragment. Error will be raised if the input Trbt does not have 48 fold symmetry.

  Parameters
  ----------
  Trbt : np.ndarray
      A three body tensor of shape (i, p, q, j, r, s, k, t, u).
  x0 : np.ndarray(3*nmodes*nmodals + nmodes*nmodals*(nmodals-1)/2), optional
      Initial guess for the parameters of the GLROv2 fragment. If None, a random guess is used. Default is None.
  tol : float
      If the L1 norm of the residual tensor is less than tol, the optimization is terminated.
  niter : int
      The maximum number of iterations for the optimization. Default is 1000.
  solver : str
      The solver to the scipy.optimize.minimize function. Default is 'BFGS'.
  return_tens : bool
      If True, the function returns the GLROv2 fragment tensors. Default is False.

  Returns
  -------
  list[np.ndarray(3, nmodes, nmodals)]
      The coefficients of the diagonalized GLROv2 fragment for each fragment. The first axis of the ndarray denotes the index of obt inside each fragment.
  list[np.ndarray(nmodes, nmodals, nmodals)]
      List of orbital rotation matrices for diagonalizing each fragment.
      The first axis of the ndarray denotes the mode index.
      The second and third axis of the ndarray denote the element of an orthogonal matrix performing the orbital rotation.
  list[np.ndarray(nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)] (Optional)
      The GLROv2 fragment tensors for each fragment
  """
  
  trbt = copy(Trbt)
  nmodes = trbt.shape[0]
  nmodals = trbt.shape[-1]
  coeff_mats = []
  rot_mats = []
  frag_tens = []


  def cost_func(x, trbt):
    frag_ten = params2_GLROv2_trbt(x, nmodes, nmodals)
    return np.sum(np.abs(trbt - frag_ten)**2)

  res_norm = np.sum(np.abs(trbt)**2)
  for ni in range (niter):
    print (f"Starting iteration {ni+1} of {niter}, residual norm: {res_norm}")
    if x0 is None:
      x0 = np.random.rand(3*nmodes*nmodals + int(nmodes*nmodals*(nmodals-1)/2))

    sol = minimize(cost_func, x0, args=(trbt,), method=solver, tol=1e-6)

    xopt = sol.x
    coeff_mat = np.reshape(xopt[:3*nmodes*nmodals], (3, nmodes, nmodals))
    rot_mat = params2rot_mat(xopt[3*nmodes*nmodals:], nmodes, nmodals)
    coeff_mats.append(coeff_mat)
    rot_mats.append(rot_mat)

    frag_ten = contract('il, jm, kn, ipl, iql, jrm, jsm, ktn, kun -> ipqjrsktu',coeff_mat[0], coeff_mat[1], coeff_mat[2],
                      rot_mat, rot_mat, rot_mat, rot_mat, rot_mat, rot_mat)
    if return_tens:
      frag_tens.append(frag_ten)

    res_norm = np.sum(np.abs(trbt - frag_ten)**2)    
    if res_norm < tol:
      break
    else:
      trbt -= frag_ten     
  print (f"Final residual norm: {res_norm}")
  if return_tens:
    return coeff_mats, rot_mats, frag_tens
  else:
    return coeff_mats, rot_mats





def params2_GLROv2_trbt(params, nmodes, nmodals):
    """
    Convert parameters defining a GLROv2 fragment to an actual three body tensor.

    Parameters
    ----------
    params : array_like
      1D array of parameters of the GLROv2 fragment. Should be of length 3 * nmodes * nmodals + nmodes * nmodals *(nmodals-1)/2.
    nmodes : int
      The number of modes.
    nmodals : int
      The number of modals.

    Returns
    -------
    np.ndarray
      The three body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals).
    """

    params1 = params[:3*nmodes*nmodals]
    coeff_mat = np.reshape(params1, (3, nmodes, nmodals))
    params2 = params[3*nmodes*nmodals:]
    rot_mat = params2rot_mat(params2, nmodes, nmodals)
    frag_ten = contract('il, jm, kn, ipl, iql, jrm, jsm, ktn, kun -> ipqjrsktu',coeff_mat[0], coeff_mat[1], coeff_mat[2],
                        rot_mat, rot_mat, rot_mat, rot_mat, rot_mat, rot_mat)
    return frag_ten














