
import numpy as np
from copy import copy
from openfermion import FermionOperator
from itertools import permutations





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
    Checks if the n body Hamiltonian tensor has n!*2^n symmetry. For example, obt has 1 symmetry, tbt has 8 symmetries, trbt (three body tensor) has 48 symmetries etc.
    (Currently only implemented up to trbt)

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
    
    if n > 3:
      raise ValueError(f"Currently only implemented upto three body tensors. Given tensor is {n} body in nature.")

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

    elif n == 3:
      def do_issymmetric(trbt, issymmetric):
        issymmetric += int(not np.allclose(trbt, np.transpose(trbt, (0, 2, 1, 3, 4, 5, 6, 7, 8))))
        issymmetric += int(not np.allclose(trbt, np.transpose(trbt, (0, 1, 2, 3, 5, 4, 6, 7, 8))))
        issymmetric += int(not np.allclose(trbt, np.transpose(trbt, (0, 1, 2, 3, 4, 5, 6, 8, 7))))
        issymmetric += int(not np.allclose(trbt, np.transpose(trbt, (0, 2, 1, 3, 5, 4, 6, 7, 8))))
        issymmetric += int(not np.allclose(trbt, np.transpose(trbt, (0, 2, 1, 3, 4, 5, 6, 8, 7))))
        issymmetric += int(not np.allclose(trbt, np.transpose(trbt, (0, 2, 1, 3, 5, 4, 6, 8, 7))))
        issymmetric += int(not np.allclose(trbt, np.transpose(trbt, (0, 1, 2, 3, 5, 4, 6, 8, 7))))
        return issymmetric
      issymmetric += do_issymmetric(nbt, issymmetric)
      nbt_perm = np.transpose(nbt, (3, 4, 5, 0, 1, 2, 6, 7, 8))
      issymmetric += do_issymmetric(nbt_perm, issymmetric)
      nbt_perm = np.transpose(nbt, (3, 4, 5, 6, 7, 8, 0, 1, 2))
      issymmetric += do_issymmetric(nbt_perm, issymmetric)
      nbt_perm = np.transpose(nbt, (6, 7, 8, 3, 4, 5, 0, 1, 2))
      issymmetric += do_issymmetric(nbt_perm, issymmetric)
      nbt_perm = np.transpose(nbt, (0, 1, 2, 6, 7, 8, 3, 4, 5))
      issymmetric += do_issymmetric(nbt_perm, issymmetric)
      nbt_perm = np.transpose(nbt, (6, 7, 8, 0, 1, 2, 3, 4, 5))
      issymmetric += do_issymmetric(nbt_perm, issymmetric)

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
        FermionOperator: Chemist ordered FermionOperator corresponding to the input two-body-tensor
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
    of the FermionOperator term ((P,1), (Q,0), (R,1), (S,0)), where P = i*nmodals + p, Q = i*nmodals + q, R = j*nmodals + r, S = j*nmodals + s.

    Args:
        tbt (np.array): two-body-tensor

    Returns:
        FermionOperator: Chemist ordered FermionOperator corresponding to the input two-body-tensor
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













def symmetrize_trbt(trbt):
  """
  Symmetrize a three body tensor to have the 48 fold symmetry.
  
  Parameters
  ----------
  trbt : np.ndarray
      A three body tensor of shape (i, p, q, j, r, s, k, t, u).
  
  Returns
  -------
  np.ndarray
      A symmetrized three body tensor of shape (i, p, q, j, r, s, k, t, u).
  """

  sym_tbt = np.zeros_like(trbt)

  M = [0, 3, 6]
  Mperms = list(permutations(M, 3))
  Ml = [(0, 1, 2), (0, 2, 1)]
  for ms in Mperms:
    for ml1 in Ml:
      for ml2 in Ml:
        for ml3 in Ml:
          perm_axes = tuple(np.add(ml1, ms[0])) + tuple(np.add(ml2, ms[1])) + tuple(np.add(ml3, ms[2]))
          sym_tbt += np.transpose(trbt, perm_axes)
  sym_tbt /= 48
  return sym_tbt










def trbt2op(trbt):
    """
    convert chemist ordered three-body-tensor to FermionOperator. The ordering convention of qubits is such that 
    an element (i, p, q, j, r, s, k, t, u) of the tensor will be mapped to the coefficient of the FermionOperator 
    term ((P,1), (Q,0), (R,1), (S,0), (T, 1), (U, 0)), where P = i*nmodals + p, Q = i*nmodals + q, R = j*nmodals + r, 
    S = j*nmodals + s, T = k*nmodals + t, U = k*nmodals + u.

    Args:
        trbt (np.array): three-body-tensor

    Returns:
        FermionOperator: Chemist ordered FermionOperator corresponding to the input three-body-tensor
    """
    nmodes = trbt.shape[0]

    op = FermionOperator()
    nmodals = trbt.shape[-1]
    for i in range(nmodes):
      for j in range(nmodes):
        for k in range(nmodes):
          for p in range(nmodals):
            P = i * nmodals + p
            for q in range(nmodals):
              Q = i * nmodals + q
              for r in range (nmodals):
                R = j * nmodals + r
                for s in range (nmodals):
                  S = j * nmodals + s
                  for t in range (nmodals):
                    T = k * nmodals + t
                    for u in range (nmodals):
                      U = k * nmodals + u
                      coeff = trbt[i,p,q,j,r,s,k,t,u]
                      if coeff != 0:
                        term = ((P,1),(Q,0), (R,1),(S,0), (T, 1), (U, 0))
                        op += FermionOperator(term, coeff)
    return op









def unperm_trbt2op(trbt):
    """
    convert unpermuted three-body-tensor to FermionOperator. The ordering convention of qubits is such that 
    an element (i, j, k, p, r, t, q, s, u) of the tensor will be mapped to the coefficient of the FermionOperator 
    term ((P,1), (Q,0), (R,1), (S,0), (T, 1), (U, 0)), where P = i*nmodals + p, Q = i*nmodals + q, R = j*nmodals + r, 
    S = j*nmodals + s, T = k*nmodals + t, U = k*nmodals + u.

    Args:
        trbt (np.array): three-body-tensor

    Returns:
        FermionOperator: Chemist ordered FermionOperator corresponding to the input three-body-tensor
    """
    nmodes = trbt.shape[0]

    op = FermionOperator()
    nmodals = trbt.shape[-1]
    for i in range(nmodes):
      for j in range(nmodes):
        for k in range(nmodes):
          for p in range(nmodals):
            P = i * nmodals + p
            for q in range(nmodals):
              Q = i * nmodals + q
              for r in range (nmodals):
                R = j * nmodals + r
                for s in range (nmodals):
                  S = j * nmodals + s
                  for t in range (nmodals):
                    T = k * nmodals + t
                    for u in range (nmodals):
                      U = k * nmodals + u
                      coeff = trbt[i,j,k,p,r,t,q,s,u]
                      if coeff != 0:
                        term = ((P,1),(Q,0), (R,1),(S,0), (T, 1), (U, 0))
                        op += FermionOperator(term, coeff)
    return op