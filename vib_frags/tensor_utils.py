import scipy as sp
import numpy as np
from copy import copy
from openfermion import QubitOperator, get_sparse_operator
from itertools import permutations
from . qubit_utils import sigdag, sig, Epq_mat, Zp_mat
from opt_einsum import contract





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







def symmetrize_tbt(tbt, force_sym = False):
  """
  Symmetrize a two body tensor to have the eight fold symmetry.
  
  Parameters
  ----------
  tbt : np.ndarray
      A two body tensor of shape (i, p, q, j, r, s).
  force_sym : bool
      If True, along with composite mode+modal index swapping symmetry, other symmetries are also enforced.
      This might be needed when dealing with small numbers and numerical instabilities.
      Default is False.
  
  Returns
  -------
  np.ndarray
      A symmetrized two body tensor of shape (i, p, q, j, r, s).
  """
  if force_sym == True:
    tbt_new = (
                tbt 
                + np.transpose(tbt, (0, 2, 1, 3, 4, 5))
                + np.transpose(tbt, (0, 1, 2, 3, 5, 4))
                + np.transpose(tbt, (0, 2, 1, 3, 5, 4))
                + np.transpose(tbt, (3, 4, 5, 0, 1, 2))
                + np.transpose(tbt, (3, 5, 4, 0, 1, 2))
                + np.transpose(tbt, (3, 4, 5, 0, 2, 1))
                + np.transpose(tbt, (3, 5, 4, 0, 2, 1))
                )/8
  else:
    tbt_new = (tbt + np.transpose(tbt, (3, 4, 5, 0, 1, 2)))/2

  return tbt_new






def obt2tbt(obt):
  '''
  promote a one-body-tensor to a two-body-tensor.

  Parameters
  ----------
  obt: np.array
    one-body tensor of shape (nmodes, nmodals, nmodals)  

  Returns
  -------
  np.array
    two-body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
  '''
  nmodes = obt.shape[0]
  nmodals = obt.shape[-1]

  tbt = np.zeros((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
  for i in range (nmodes):
     obt_i = obt[i]
     e_i, u_i = np.linalg.eigh(obt_i)
     tbt[i,:,:,i,:,:] = np.einsum('k, pk, qk, rk, sk -> pqrs', e_i, u_i, u_i, u_i, u_i, optimize = True)

  return tbt







def obt2op(obt):
  '''
  convert one-body-tensor to QubitOperator. The ordering convention of qubits is such that an element (i, p, q) of the tensor will be mapped to the coefficient
  of the QubitOperator term ((P,1), (Q,0)), where P = i*nmodals + p, Q = i*nmodals + q.

  Parameters
  ----------
  obt: np.array
    one-body tensor of shape (nmodes, nmodals, nmodals)  

  Returns
  -------
  QubitOperator
    Chemist ordered QubitOperator corresponding to the input one-body-tensor
  '''
  nmodes = obt.shape[0]
  nmodals = obt.shape[-1]
  op = QubitOperator()

  for i in range (nmodes):
    for p in range (nmodals):
      P = i * nmodals + p
      for q in range (nmodals):
        Q = i * nmodals + q
        coeff = obt[i,p,q]
        if coeff != 0:
          op += coeff * sigdag(P) * sig(Q)
  return op







def tbt2op(tbt):
    """
    convert two-body-tensor to QubitOperator. The ordering convention of qubits is such that an element (i, p, q, j, r, s) of the tensor will be mapped to the coefficient
    of the QubitOperator term ((P,1), (Q,0), (R,1), (S,0)), where P = i*nmodals + p, Q = i*nmodals + q, R = j*nmodals + r, S = j*nmodals + s.

    Args:
        tbt (np.array): two-body-tensor

    Returns:
        QubitOperator: Chemist ordered QubitOperator corresponding to the input two-body-tensor
    """
    nmodes = tbt.shape[0]

    op = QubitOperator()
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
                coeff = tbt[i,p,q,j,r,s]
                if coeff != 0:
                  op += coeff * sigdag(P) * sig(Q) * sigdag(R) * sig(S)
    return op







def unperm_tbt2op(tbt):
    """
    convert unpermuted two-body-tensor to QubitOperator. The ordering convention of qubits is such that an element (i, j, p, r, q, s) of the tensor will be mapped to the coefficient
    of the QubitOperator term ((P,1), (Q,0), (R,1), (S,0)), where P = i*nmodals + p, Q = i*nmodals + q, R = j*nmodals + r, S = j*nmodals + s.

    Args:
        tbt (np.array): two-body-tensor

    Returns:
        QubitOperator: Chemist ordered QubitOperator corresponding to the input two-body-tensor
    """
    nmodes = tbt.shape[0]

    op = QubitOperator()
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
                coeff = tbt[i,j,p,r,q,s]
                if coeff != 0:
                  op += coeff * sigdag(P) * sig(Q) * sigdag(R) * sig(S)
    return op













def symmetrize_trbt(trbt, force_sym = False):
  """
  Symmetrize a three body tensor to have the 48 fold symmetry.
  
  Parameters
  ----------
  trbt : np.ndarray
      A three body tensor of shape (i, p, q, j, r, s, k, t, u).
  force_sym : bool
      If True, along with composite mode+modal index swapping symmetry, other symmetries are also enforced.
      This might be needed when dealing with small numbers and numerical instabilities.
      Default is False.
      
  Returns
  -------
  np.ndarray
      A symmetrized three body tensor of shape (i, p, q, j, r, s, k, t, u).
  """

  sym_tbt = np.zeros_like(trbt)

  M = [0, 3, 6]
  Mperms = list(permutations(M, 3))
  if force_sym == True:
    Ml = [(0, 1, 2), (0, 2, 1)]
    for ms in Mperms:
      for ml1 in Ml:
        for ml2 in Ml:
          for ml3 in Ml:
            perm_axes = tuple(np.add(ml1, ms[0])) + tuple(np.add(ml2, ms[1])) + tuple(np.add(ml3, ms[2]))
            sym_tbt += np.transpose(trbt, perm_axes)
    sym_tbt /= 48
  else:
    mls = (0, 1, 2)
    for ms in Mperms:
      perm_axes = tuple(np.add(mls, ms[0])) + tuple(np.add(mls, ms[1])) + tuple(np.add(mls, ms[2]))
      sym_tbt += np.transpose(trbt, perm_axes)
    sym_tbt /= 6
  
  return sym_tbt










def trbt2op(trbt):
    """
    convert chemist ordered three-body-tensor to QubitOperator. The ordering convention of qubits is such that 
    an element (i, p, q, j, r, s, k, t, u) of the tensor will be mapped to the coefficient of the QubitOperator 
    term ((P,1), (Q,0), (R,1), (S,0), (T, 1), (U, 0)), where P = i*nmodals + p, Q = i*nmodals + q, R = j*nmodals + r, 
    S = j*nmodals + s, T = k*nmodals + t, U = k*nmodals + u.

    Args:
        trbt (np.array): three-body-tensor

    Returns:
        QubitOperator: Chemist ordered QubitOperator corresponding to the input three-body-tensor
    """
    nmodes = trbt.shape[0]

    op = QubitOperator()
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
                        op += coeff * sigdag(P) * sig(Q) * sigdag(R) * sig(S) * sigdag(T) * sig(U)
    return op









def unperm_trbt2op(trbt):
    """
    convert unpermuted three-body-tensor to QubitOperator. The ordering convention of qubits is such that 
    an element (i, j, k, p, r, t, q, s, u) of the tensor will be mapped to the coefficient of the QubitOperator 
    term ((P,1), (Q,0), (R,1), (S,0), (T, 1), (U, 0)), where P = i*nmodals + p, Q = i*nmodals + q, R = j*nmodals + r, 
    S = j*nmodals + s, T = k*nmodals + t, U = k*nmodals + u.

    Args:
        trbt (np.array): three-body-tensor

    Returns:
        QubitOperator: Chemist ordered QubitOperator corresponding to the input three-body-tensor
    """
    nmodes = trbt.shape[0]

    op = QubitOperator()
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
                        op += coeff * sigdag(P) * sig(Q) * sigdag(R) * sig(S) * sigdag(T) * sig(U) 
    return op












def obt2_proj_mat(obt):
  '''
  Map one-body-tensor to scipy sparse matrix representation of corresponding one-body opeartor in the subspace of one excitation per mode.

  Parameters
  ----------
  obt: np.array
    one-body tensor of shape (nmodes, nmodals, nmodals)  

  Returns
  -------
  sp.sparse.csc_matrix
    sparse matrix of one-body operator in the subspace of one excitation per mode.
  '''
  nmodes = obt.shape[0]
  nmodals = obt.shape[-1]
  
  op = 0
  for i in range (nmodes):
    for p in range (nmodals):
      for q in range (nmodals):
        coeff = obt[i,p,q]
        if coeff != 0:
          op += coeff * Epq_mat(i, p, q, nmodals, nmodes)
  return op







def tbt2_proj_mat(tbt):
    """
    Map two-body-tensor to scipy sparse matrix representation of corresponding two-body opeartor in the subspace of one excitation per mode.

    Args:
        tbt (np.array): two-body-tensor

    Returns:
        sp.sparse.csc_matrix: sparse matrix of two-body operator in the subspace of one excitation per mode.
    """
    nmodes = tbt.shape[0]

    op = 0
    nmodals = tbt.shape[2]
    for i in range(nmodes):
      for j in range(nmodes):
        for p in range(nmodals):
          for q in range(nmodals):
            for r in range (nmodals):
              for s in range (nmodals):
                coeff = tbt[i,p,q,j,r,s]
                if coeff != 0:
                  op += coeff * Epq_mat(i, p, q, nmodals, nmodes) * Epq_mat(j, r, s, nmodals, nmodes)
    return op







def unperm_tbt2_proj_mat(tbt):
    """
    Map unpermuted two-body-tensor to scipy sparse matrix representation of corresponding two-body opeartor in the subspace of one excitation per mode.

    Args:
        tbt (np.array): two-body-tensor

    Returns:
        sp.sparse.csc_matrix: sparse matrix of two-body operator in the subspace of one excitation per mode.
    """
    nmodes = tbt.shape[0]

    op = 0
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
                coeff = tbt[i,j,p,r,q,s]
                if coeff != 0:
                  op += coeff * Epq_mat(i, p, q, nmodals, nmodes) * Epq_mat(j, r, s, nmodals, nmodes)
    return op










def trbt2_proj_mat(trbt):
    """
    Map three-body-tensor to scipy sparse matrix representation of corresponding three-body opeartor in the subspace of one excitation per mode.

    Args:
        trbt (np.array): three-body-tensor

    Returns:
        sp.sparse.csc_matrix: sparse matrix of three-body operator in the subspace of one excitation per mode.
    """
    nmodes = trbt.shape[0]

    op = 0
    nmodals = trbt.shape[-1]
    for i in range(nmodes):
      for j in range(nmodes):
        for k in range(nmodes):
          for p in range(nmodals):
            for q in range(nmodals):
              for r in range (nmodals):
                for s in range (nmodals):
                  for t in range (nmodals):
                    for u in range (nmodals):
                      coeff = trbt[i,p,q,j,r,s,k,t,u]
                      if coeff != 0:
                        op += coeff * Epq_mat(i, p, q, nmodals, nmodes) * Epq_mat(j, r, s, nmodals, nmodes) * Epq_mat(k, t, u, nmodals, nmodes)
    return op









def unperm_trbt2_proj_mat(trbt):
    """
    Map unpermuted three-body-tensor to scipy sparse matrix representation of corresponding three-body opeartor in the subspace of one excitation per mode.

    Args:
        trbt (np.array): three-body-tensor

    Returns:
        sp.sparse.csc_matrix: sparse matrix of three-body operator in the subspace of one excitation per mode.
    """
    nmodes = trbt.shape[0]

    op = 0
    nmodals = trbt.shape[-1]
    for i in range(nmodes):
      for j in range(nmodes):
        for k in range(nmodes):
          for p in range(nmodals):
            for q in range(nmodals):
              for r in range (nmodals):
                for s in range (nmodals):
                  for t in range (nmodals):
                    for u in range (nmodals):
                      coeff = trbt[i,j,k,p,r,t,q,s,u]
                      if coeff != 0:
                        op += coeff * Epq_mat(i, p, q, nmodals, nmodes) * Epq_mat(j, r, s, nmodals, nmodes) * Epq_mat(k, t, u, nmodals, nmodes)
    return op













def SO_2_sarray(SO, nmodals, nmodes):
    """
    Convert a special orthogonal matrix SO to a scipy sparse array. Ensure that SO[i] belongs to SO(n) i.e. its determinant is +1 and not -1.
    To ensure this, if O[i] is an orthogonal matrix with det(O[i]) = -1, set O[i, :, -1] *= -1.

    Parameters
    ----------
    SO : np.ndarray
        Special orthogonal matrix of shape (nmodes, nmodals, nmodals).
    nmodals : int
        Number of modals.
    nmodes : int
        Number of modes.

    Returns
    -------
    sp.sparse.csc_matrix
        Full space unitary as scipy sparse array corresponding to the orthogonal matrix O
    """
    kappa_hat = QubitOperator()
    for i in range (nmodes):
        if np.linalg.det(SO[i]) < 0:
           raise ValueError("Given orbital rotation tensor has a matrix that does not belong to SO(n). \nTo correct this, if O[i] is an orthogonal matrix with det(O[i]) = -1, set O[i, :, -1] *= -1.")
        kappa = sp.linalg.logm(SO[i])
        for p in range (nmodals):
            p_i = i*nmodals + p
            for q in range (nmodals):
                q_i = i*nmodals + q
                kappa_hat += kappa[p, q]*sigdag(p_i)*sig(q_i)
    kappa_hat_sarray = get_sparse_operator(kappa_hat, n_qubits=nmodes*nmodals)
    U = sp.sparse.linalg.expm(kappa_hat_sarray)
    return sp.sparse.csc_matrix(U)










def SO_2_proj_mat(SO, nmodals, nmodes):
    """
    Get the sparse matrix representation of the opeartor corresnpoding to a special orthogonal matrix O, in the projected subspace of one excitation per mode.
    To ensure this, if O[i] is an orthogonal matrix with det(O[i]) = -1, set O[i, :, -1] *= -1.

    Parameters
    ----------
    SO : np.ndarray
        Special orthogonal matrix of shape (nmodes, nmodals, nmodals).
    nmodals : int
        Number of modals.
    nmodes : int
        Number of modes.

    Returns
    -------
    sp.sparse.csc_matrix
        Sparse matrix representation of the opeartor corresnpoding to an orthogonal matrix O in the projected subspace of one excitation per mode.
    """
    kappa_mat = 0
    for i in range (nmodes):
        if np.linalg.det(SO[i]) < 0:
           raise ValueError("Given orbital rotation tensor has a matrix that does not belong to SO(n). \nTo correct this, if O[i] is an orthogonal matrix with det(O[i]) = -1, set O[i, :, -1] *= -1.")
        kappa = np.real(sp.linalg.logm(SO[i]))
        for p in range (nmodals):
            for q in range (nmodals):
                kappa_mat += kappa[p, q]*Epq_mat(i, p, q, nmodals, nmodes)
    U = sp.sparse.linalg.expm(kappa_mat)
    return sp.sparse.csc_matrix(U)









def refcart_2_Qop(tensor):
    '''
    Convert a coefficient tensor of obt, tbt or trbt cartans written in terms of reflections to qubit operator.

    Parameters
    ----------
    tensor : np.ndarray
        Coefficient tensor of tbt or trbt cartans written in terms of reflections. Accpeted dimensions an ordering of axes are as follows:
        - For one-body tensor: (nmodes, nmodals, nmodals) or (nmodes, nmodals)
        - For two-body tensor: (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
        - For three-body tensor: (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
        If the tensors are of lower rank due to redunduncies, promote them to full rank and then pass them to the funciton.
    Returns
    -------
    openfermion.QubitOperator
        QubitOperator built from the coefficient tensor of tbt or trbt cartans written in terms of reflections.
    '''
    if tensor.ndim == 2:  # One-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        qubit_op = QubitOperator()
        for i in range(nmodes):
            for l in range(nmodals):
                coeff = tensor[i, l]
                if coeff != 0:
                    qubit_op += coeff * QubitOperator(f'Z{i*nmodals + l}')
    elif tensor.ndim == 3:  # One-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        qubit_op = QubitOperator()
        for i in range(nmodes):
            for l in range(nmodals):
                coeff = tensor[i, l, l]
                if coeff != 0:
                    qubit_op += coeff * QubitOperator(f'Z{i*nmodals + l}')
    elif tensor.ndim == 6:  # Two-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        qubit_op = QubitOperator()
        for i in range(nmodes):
            for j in range(i):
                for l in range(nmodals):
                    for m in range(nmodals):
                        coeff = tensor[i, l, l, j, m, m]
                        if coeff != 0:
                            qubit_op += coeff * QubitOperator(f'Z{i*nmodals + l} Z{j*nmodals + m}')
    elif tensor.ndim == 9:  # Three-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        qubit_op = QubitOperator()
        for i in range(nmodes):
            for j in range(i):
                for k in range(j):
                    for l in range(nmodals):
                        for m in range(nmodals):
                            for n in range(nmodals):
                                coeff = tensor[i, l, l, j, m, m, k, n, n]
                                if coeff != 0:
                                    qubit_op += coeff * QubitOperator(f'Z{i*nmodals + l} Z{j*nmodals + m} Z{k*nmodals + n}')
    else:
        raise ValueError("Input tensor must be either a onr-body tensor (2D or 3D), two-body tensor (6D) or a three-body tensor (9D).")
    return qubit_op









def unperm_refcart_2_Qop(tensor):
    '''
    Convert a coefficient tensor of obt, unpermuted tbt or unpermuted trbt cartans written in terms of reflections to qubit operator.

    Parameters
    ----------
    tensor : np.ndarray
        Coefficient tensor of tbt or trbt cartans written in terms of reflections. Accpeted dimensions an ordering of axes are as follows:
        - For one-body tensor: (nmodes, nmodals, nmodals) or (nmodes, nmodals)
        - For two-body tensor: (nmodes, nmodes, nmodals, nmodals, nmodals, nmodals)
        - For three-body tensor: (nmodes, nmodes, nmodes, nmodals, nmodals, nmodals, nmodals, nmodals, nmodals)
        If the tensors are of lower rank due to redunduncies, promote them to full rank and then pass them to the funciton.
    Returns
    -------
    openfermion.QubitOperator
        QubitOperator built from the coefficient tensor of tbt or trbt cartans written in terms of reflections.
    '''
    if tensor.ndim == 2:  # One-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        qubit_op = QubitOperator()
        for i in range(nmodes):
            for l in range(nmodals):
                coeff = tensor[i, l]
                if coeff != 0:
                    qubit_op += coeff * QubitOperator(f'Z{i*nmodals + l}')
    elif tensor.ndim == 3:  # One-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        qubit_op = QubitOperator()
        for i in range(nmodes):
            for l in range(nmodals):
                coeff = tensor[i, l, l]
                if coeff != 0:
                    qubit_op += coeff * QubitOperator(f'Z{i*nmodals + l}')
    elif tensor.ndim == 6:  # Two-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        qubit_op = QubitOperator()
        for i in range(nmodes):
            for j in range(i):
                for l in range(nmodals):
                    for m in range(nmodals):
                        coeff = tensor[i, j, l, m, l, m]
                        if coeff != 0:
                            qubit_op += coeff * QubitOperator(f'Z{i*nmodals + l} Z{j*nmodals + m}')
    elif tensor.ndim == 9:  # Three-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        qubit_op = QubitOperator()
        for i in range(nmodes):
            for j in range(i):
                for k in range(j):
                    for l in range(nmodals):
                        for m in range(nmodals):
                            for n in range(nmodals):
                                coeff = tensor[i, j, k, l, m, n, l, m, n]
                                if coeff != 0:
                                    qubit_op += coeff * QubitOperator(f'Z{i*nmodals + l} Z{j*nmodals + m} Z{k*nmodals + n}')
    else:
        raise ValueError("Input tensor must be either a onr-body tensor (2D or 3D), two-body tensor (6D) or a three-body tensor (9D).")
    return qubit_op











def refcart_2_proj_mat(tensor):
    '''
    Convert obt, tbt or trbt cartans assumed to be coeffients of reflections to sparse operator in the projected subspace of one excitation per mode.

    Parameters
    ----------
    tensor : np.ndarray
        Coefficient tensor of tbt or trbt cartans written in terms of reflections. Accpeted dimensions an ordering of axes are as follows:
        - For one-body tensor: (nmodes, nmodals, nmodals) or (nmodes, nmodals)
        - For two-body tensor: (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
        - For three-body tensor: (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
        If the tensors are of lower rank due to redunduncies, promote them to full rank and then pass them to the funciton.
    Returns
    -------
    sp.sparse.csc_matrix
        Sparse array representing the operator constructed from reflection cartan tensor in the subsace of one excitation per mode.
    '''
    if tensor.ndim == 2:  # One-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        sparse_mat = 0
        for i in range(nmodes):
            for l in range(nmodals):
                coeff = tensor[i, l]
                if coeff != 0:
                    sparse_mat += coeff * Zp_mat(i, l, nmodals, nmodes)
    elif tensor.ndim == 3:  # One-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        sparse_mat = 0
        for i in range(nmodes):
            for l in range(nmodals):
                coeff = tensor[i, l, l]
                if coeff != 0:
                    sparse_mat += coeff * Zp_mat(i, l, nmodals, nmodes)
    elif tensor.ndim == 6:  # Two-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        sparse_mat = 0
        for i in range(nmodes):
            for j in range(i):
                for l in range(nmodals):
                    for m in range(nmodals):
                        coeff = tensor[i, l, l, j, m, m]
                        if coeff != 0:
                            sparse_mat += coeff * Zp_mat(i, l, nmodals, nmodes) * Zp_mat(j, m, nmodals, nmodes)
    elif tensor.ndim == 9:  # Three-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        sparse_mat = 0
        for i in range(nmodes):
            for j in range(i):
                for k in range(j):
                    for l in range(nmodals):
                        for m in range(nmodals):
                            for n in range(nmodals):
                                coeff = tensor[i, l, l, j, m, m, k, n, n]
                                if coeff != 0:
                                    sparse_mat += coeff * Zp_mat(i, l, nmodals, nmodes) * Zp_mat(j, m, nmodals, nmodes) * Zp_mat(k, n, nmodals, nmodes)
    else:
        raise ValueError("Input tensor must be either a one-body tensor (2D or 3D), two-body tensor (6D) or a three-body tensor (9D).")
    return sparse_mat












def unperm_refcart_2_proj_mat(tensor):
    '''
    Convert obt, unpermuted tbt or unpermuted trbt cartans assumed to be coeffients of reflections to sparse operator in the projected subspace of one excitation per mode.

    Parameters
    ----------
    tensor : np.ndarray
        Coefficient tensor of tbt or trbt cartans written in terms of reflections. Accpeted dimensions an ordering of axes are as follows:
        - For one-body tensor: (nmodes, nmodals, nmodals) or (nmodes, nmodals)
        - For two-body tensor: (nmodes, nmodes, nmodals, nmodals, nmodals, nmodals)
        - For three-body tensor: (nmodes, nmodes, nmodes, nmodals, nmodals, nmodals, nmodals, nmodals, nmodals)
        If the tensors are of lower rank due to redunduncies, promote them to full rank and then pass them to the funciton.
    Returns
    -------
    sp.sparse.csc_matrix
        Sparse array representing the operator constructed from reflection cartan tensor in the subsace of one excitation per mode.
    '''
    if tensor.ndim == 2:  # One-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        sparse_mat = 0
        for i in range(nmodes):
            for l in range(nmodals):
                coeff = tensor[i, l]
                if coeff != 0:
                    sparse_mat += coeff * Zp_mat(i, l, nmodals, nmodes)
    elif tensor.ndim == 3:  # One-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        sparse_mat = 0
        for i in range(nmodes):
            for l in range(nmodals):
                coeff = tensor[i, l, l]
                if coeff != 0:
                    sparse_mat += coeff * Zp_mat(i, l, nmodals, nmodes)
    elif tensor.ndim == 6:  # Two-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        sparse_mat = 0
        for i in range(nmodes):
            for j in range(i):
                for l in range(nmodals):
                    for m in range(nmodals):
                        coeff = tensor[i, j, l, m, l, m]
                        if coeff != 0:
                            sparse_mat += coeff * Zp_mat(i, l, nmodals, nmodes) * Zp_mat(j, m, nmodals, nmodes)
    elif tensor.ndim == 9:  # Three-body tensor
        nmodes, nmodals = tensor.shape[0], tensor.shape[-1]
        sparse_mat = 0
        for i in range(nmodes):
            for j in range(i):
                for k in range(j):
                    for l in range(nmodals):
                        for m in range(nmodals):
                            for n in range(nmodals):
                                coeff = tensor[i, j, k, l, m, n, l, m, n]
                                if coeff != 0:
                                    sparse_mat += coeff * Zp_mat(i, l, nmodals, nmodes) * Zp_mat(j, m, nmodals, nmodes) * Zp_mat(k, n, nmodals, nmodes)
    else:
        raise ValueError("Input tensor must be either a one-body tensor (2D or 3D), two-body tensor (6D) or a three-body tensor (9D).")
    return sparse_mat








def get_opt_Pauli_LCU_tensors(C = None, obt = None, tbt = None, trbt = None):
    '''
    Get the modified tensors obtained by contracting tensors in sub-optimal Pauli LCU due to redundunt identities. 
    This gives the input to the function "qubit_utils.get_QOP_of_Pauli_LCU" when opt=True.
    We assume atleast one of the three tensors are provided.
    
    parameters
    ----------
    C : int (optional)
        Constant term
    obt: np.array (optional)
      one-body tensor of the Hamiltonian of shape (nmodes, nmodals, nmodals).
    tbt: np.array (optional)
      two-body tensor of the Hamiltonian of shape (nmodes, nmodals, nmodals,nmodes, nmodals, nmodals)
    trbt: np.array (optional)
      three-body tensor of the Hamiltonian of shape (nmodes, nmodals, nmodals,nmodes, nmodals, nmodals,nmodes, nmodals, nmodals)  
    
    Returns
    -------
    openfermion.QubitOperator
        QubitOperator corresponding to the new constant term
    openfermion.QubitOperator
        QubitOperator corresponding to the new obt
    openfermion.QubitOperator
        QubitOperator corresponding to the new tbt
    openfermion.QubitOperator
        QubitOperator corresponding to the new trbt
    '''
    
    if all(ten is None for ten in (obt, tbt, trbt)):
       raise TypeError('Atleast one of the tensors should not be None')

    nmodes = next(ten.shape[0] for ten in (obt, tbt, trbt) if ten is not None)
    nmodals = next(ten.shape[-1] for ten in (obt, tbt, trbt) if ten is not None)
    
    if C is None:
       C = 0
    if obt is None:
       obt = np.zeros((nmodes, nmodals, nmodals))
    if tbt is None:
       tbt = np.zeros((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
    if trbt is None:
       trbt = np.zeros((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
    tbt_sym = symmetrize_tbt(tbt)
    trbt_sym = symmetrize_trbt(trbt)

    C_tilde = C + contract('ipp -> ', obt)/2 + contract('ippjrr -> ', tbt_sym)/4 + contract('ippjrrktt -> ', trbt_sym)/8
    obt_tilde = obt + contract('ipqjrr -> ipq', tbt_sym) + contract('ipqjrrktt -> ipq', trbt_sym)*3/4
    tbt_tilde = tbt_sym + contract('ipqjrsktt -> ipqjrs', trbt_sym)*3/2
    trbt_tilde = trbt_sym

    return C_tilde, obt_tilde, tbt_tilde, trbt_tilde
    






def opt_Pauli_LCU_1norm(obt,tbt,trbt,ret_const=False):
    '''
    Obtian the induced one-norm of the optimal Pauli LCU for the Hamiltonian defined by the given tensors. Refer to the overleaf document for more information.

    Parameters
    ----------
    obt: np.array
      one-body tensor of shape (nmodes, nmodals, nmodals) 
    tbt: np.array
      two-body tensor of shape (nmodes, nmodals, nmodals,nmodes, nmodals, nmodals)  
    trbt: np.array
      three-body tensor of shape (nmodes, nmodals, nmodals,nmodes, nmodals, nmodals,nmodes, nmodals, nmodals)  
    ret_const: bool
      If true, returns the coefficient of the identity in the Pauli LCU

    Returns
    -------
    float
      one-norm of the Pauli LCU
    float (if ret_const = True)
      coefficient of the identity in the Pauli LCU
    '''

    C_tilde, obt_tilde, tbt_tilde, trbt_tilde = get_opt_Pauli_LCU_tensors(C = None, obt = obt, tbt = tbt, trbt = trbt)

    one_norm = np.sum(np.abs(obt_tilde))/2 + np.sum(np.abs(tbt_tilde))/4 + np.sum(np.abs(trbt_tilde))/8

    if ret_const:
       return one_norm, C_tilde
    else:
      return one_norm
    

















def get_opt_THC_LCU_tensors(C = None, obt = None, tbt = None, trbt = None, zeta = None, gamma = None):
    '''
    Get the modified tensors obtained by contracting THC tensors due to changing number operators to reflections.
    We assume atleast one of the three tensors are provided.
    
    parameters
    ----------
    C : int (optional)
        Constant term
    obt: np.array (optional)
      one-body tensor of the Hamiltonian of shape (nmodes, nmodals, nmodals).
    tbt: np.array (optional)
      Fully symmetrized two-body tensor of the Hamiltonian of shape (nmodes, nmodals, nmodals,nmodes, nmodals, nmodals)
    trbt: np.array (optional)
      Fully symmetrized three-body tensor of the Hamiltonian of shape (nmodes, nmodals, nmodals,nmodes, nmodals, nmodals,nmodes, nmodals, nmodals)  
    zeta: np.array (optional)
      THC two-mode tensor of shape (nmodes, nmodes, nthc, nthc)
    gamma: np.array (optional)
      THC three-mode tensor of shape (nmodes, nmodes, nmodes, nthc, nthc, nthc)

    
    Returns
    -------
        Modified tensors
    '''
    
    if all(ten is None for ten in (obt, tbt, trbt)):
       raise TypeError('Atleast one of the Hamiltonian tensors should not be None')
    
    if all(ten is None for ten in (zeta, gamma)):
       raise TypeError('Atleast one of the THC tensors should not be None')

    nmodes = next(ten.shape[0] for ten in (obt, tbt, trbt) if ten is not None)
    nmodals = next(ten.shape[-1] for ten in (obt, tbt, trbt) if ten is not None)
    nthc = next(ten.shape[-1] for ten in (zeta, gamma) if ten is not None)
    
    if C is None:
       C = 0
    if obt is None:
       obt = np.zeros((nmodes, nmodals, nmodals))
    if tbt is None:
       tbt = np.zeros((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
    if trbt is None:
       trbt = np.zeros((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
    if zeta is None:
       zeta = np.zeros((nmodes, nmodes, nthc, nthc))
    if gamma is None:
       gamma = np.zeros((nmodes, nmodes, nmodes, nthc, nthc, nthc))

    gamma_tilde = -gamma
    zeta_tilde = zeta + 3*contract('ijkuvw -> ijuv', gamma)
    obt_tilde = obt - (1/2)*contract('ipqjrr -> ipq', tbt) - (3/8)*contract('ipqjrrktt -> ipq', trbt)
    C_tilde = C + (1/2)*contract('ipp -> ', obt_tilde) + (1/4)*contract('ippjrr -> ', tbt) + (1/8)*contract('ippjrrktt -> ', trbt)

    return C_tilde, obt_tilde, zeta_tilde, gamma_tilde