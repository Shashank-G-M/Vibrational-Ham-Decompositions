
import numpy as np
from copy import copy
from openfermion import FermionOperator






def transpose_nbt(nbt):
    """
    Permutes axes of an n body tensor from (i, j, ..., p, r, ..., q, s, ...) to (i, p, q, j, r, s, ...). i, j index modes, 
    p, r index creation operators on modals p and r, and q, s index annihilation operators on modals q and s.
    This permutation is necessary to reshape the tensor into a matrix that can be decomposed into fragments.
    Currently implemented only for one, two and three body tensors.

    Parameters
    ----------
    nbt : np.ndarray
        An n body tensor of shape (i, j, ..., p, r, ..., q, s, ...).

    Returns
    -------
    np.ndarray
        An n body tensor of shape (i, p, q, j, r, s, ...).
    """

    # Get the bodiness of the tensor
    n = len(nbt.shape) // 3

    if n == 1:
      return nbt
    elif n == 2:
      new_nbt = np.transpose(nbt, (0, 2, 4, 1, 3, 5))
    elif n == 3:
      new_nbt = np.transpose(nbt, (0, 3, 6, 1, 4, 7, 2, 5, 8))
    else:
      raise ValueError("Currently only implemented for one, two and three body tensors.")

    return new_nbt







def check_symmetry(nbt):
    """
    Checks if the n body Hamiltonian tensor has n!*2^n symmetry. For example, obt has 1 symmetry, tbt has 8 symmetries, rbt (three body tensor) has 48 symmetries etc.
    (Currently only implemented for obt and tbt)

    Parameters
    ----------
    nbt : np.ndarray
        An n body tensor of shape (i, p, q, j, r, s, k, t, u, ...). General structure is, mode indices are followed by modal indices of that mode.

    Returns
    -------
    bool
        True if the tensor has correct symmetry, False otherwise.
    """
    # Get the bodiness of the tensor
    n = len(nbt.shape) // 3
    
    if n > 2:
      raise ValueError(f"Currently only implemented for one and two body tensors. Given tensor is {n} body in nature.")

    issymmetric = 0
    if n == 1:
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (0, 2, 1))))
    
    elif n == 2:
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (0, 2, 1, 3, 4, 5))))
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (0, 1, 2, 3, 5, 4))))
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (0, 2, 1, 3, 5, 4))))
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (3, 4, 5, 0, 1, 2))))
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (3, 5, 4, 0, 1, 2))))
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (3, 4, 5, 0, 2, 1))))
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (3, 5, 4, 0, 2, 1))))

    return issymmetric == 0







def symmetrize_tbt(tbt):
  """
  Symmetrize a two body tensor to have the eight fold symmetry.
  
  Parameters
  ----------
  tbt : np.ndarray
      A two body tensor of shape (i, p, q, j, r, s).
  
  Returns
  -------
  np.ndarray
      A symmetrized two body tensor of shape (i, p, q, j, r, s).
  """
  tbt_copy = copy(tbt)

  tbt_copy = (
              tbt_copy 
              + np.transpose(tbt_copy, (0, 2, 1, 3, 4, 5))
              + np.transpose(tbt_copy, (0, 1, 2, 3, 5, 4))
              + np.transpose(tbt_copy, (0, 2, 1, 3, 5, 4))
              + np.transpose(tbt_copy, (3, 4, 5, 0, 1, 2))
              + np.transpose(tbt_copy, (3, 5, 4, 0, 1, 2))
              + np.transpose(tbt_copy, (3, 4, 5, 0, 2, 1))
              + np.transpose(tbt_copy, (3, 5, 4, 0, 2, 1))
              )/8

  return tbt_copy






def tbt2op(tbt):
    """
    convert two-body-tensor to FermionOperator. The ordering convention of qubits is such that an element (i, p, q, j, r, s) of the tensor will be mapped to the coefficient
    of the FermionOperator term ((P,1), (Q,0), (R,1), (S,0)), where P = i*nmodes + p, Q = i*nmodes + q, R = j*nmodes + r, S = j*nmodes + s.

    Args:
        tbt (np.array): two-body-tensor

    Returns:
        FermionOperator: FermionOperator corresponding to the input chemist ordered two-body-tensor
    """
    nmodes = tbt.shape[0]

    op = FermionOperator()
    nmodals = tbt.shape[2]
    for i in range(nmodes):
      for j in range(nmodes):
        for p in range(nmodals):
          P = i * nmodals + p
          for q in range(nmodals):
            Q = i * nmodals + q
            for r in range (nmodals):
              R = j * nmodals + r
              for s in range (nmodals):
                S = j * nmodals + s
                term = ((P,1),(Q,0), (R,1),(S,0))
                coeff = tbt[i,p,q,j,r,s]
                op += FermionOperator(term, coeff)
    return op











def unperm_tbt2op(tbt):
    """
    convert unpermuted two-body-tensor to FermionOperator. The ordering convention of qubits is such that an element (i, j, p, r, q, s) of the tensor will be mapped to the coefficient
    of the FermionOperator term ((P,1), (Q,0), (R,1), (S,0)), where P = i*nmodes + p, Q = i*nmodes + q, R = j*nmodes + r, S = j*nmodes + s.

    Args:
        tbt (np.array): two-body-tensor

    Returns:
        FermionOperator: FermionOperator corresponding to the input chemist ordered two-body-tensor
    """
    nmodes = tbt.shape[0]

    op = FermionOperator()
    nmodals = tbt.shape[2]
    for i in range(nmodes):
      for j in range(nmodes):
        for p in range(nmodals):
          P = i * nmodals + p
          for q in range(nmodals):
            Q = i * nmodals + q
            for r in range (nmodals):
              R = j * nmodals + r
              for s in range (nmodals):
                S = j * nmodals + s
                term = ((P,1),(Q,0), (R,1),(S,0))
                coeff = tbt[i,j,p,r,q,s]
                op += FermionOperator(term, coeff)
    return op
