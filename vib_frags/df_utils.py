#Utility file to perfeorm double factorization of two mode vibrational Hamiltonian in Christiansen form
import numpy as np
from . import tensor_utils as tu

def SF_terms(tbt):
    """
    Computes the single factorization of a two body tensor. The tensor is reshaped into a matrix of shape (i*p*q, j*r*s) and the eigenvalues and eigenvectors are returned.
    Error will be raised if the input tbt does not have eight fold symmetry.

    Parameters
    ----------
    tbt : np.ndarray
        A two body tensor of shape (i, p, q, j, r, s).

    Returns
    -------
    np.ndarray
        Eigenvalues of the reshaped tensor.
    np.ndarray
        Eigenvectors of the reshaped tensor.
    """
    # Check if the tensor has the correct symmetry
    if not tu.check_symmetry(tbt):
      raise ValueError("The tensor does not have the correct symmetry.")
    
    # Compute the single factorization of the tensor
    tbt_mat = np.reshape(tbt, (tbt.shape[0] * tbt.shape[1] * tbt.shape[2], tbt.shape[3] * tbt.shape[4] * tbt.shape[5]))
    
    # Compute the eigenvalues and eigenvectors of the reshaped tensor
    eigvals, eigvecs = np.linalg.eigh(tbt_mat)
    
    return eigvals, eigvecs







