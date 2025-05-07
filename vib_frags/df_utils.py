#Utility file to perfeorm double factorization of two mode vibrational Hamiltonian in Christiansen form
import numpy as np
from . import tensor_utils as tu

def SF_terms(tbt):
    """
    Computes single factorization of a two body tensor. The tensor is reshaped into a matrix of shape (i*p*q, j*r*s) and the eigenvalues and eigenvectors are returned.
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





def DF_terms(tbt):
    """
    Computes double factorization of a two body tensor. Returns three objects: Eigenvalues from first factorization sorted according to their magnitude, 
    Eigenvalues from second factorization sorted according to their magnitude encoded in a tensor, and the orbital rotation matrix for each node of each 
    fragment encoded in a tensor.
    Error will be raised if the input tbt does not have eight fold symmetry.

    Parameters
    ----------
    tbt : np.ndarray
        A two body tensor of shape (i, p, q, j, r, s).

    Returns
    -------
    np.ndarray (nfrags,)
        Eigenvalues from first factorization sorted according to their magnitude.
    np.ndarray (nfrags, nmodes, nmodals)
        Eigenvalues from second factorization for each fragment and node sorted according to their magnitude.
    np.ndarray (nfrags, nmodes, nmodals, nmodals)
        Orbital rotation matrix for each node of each fragment.
        The first axis denotes the fragment index.
        The second axis denotes the mode index.
        The third and fourth axis denote the element of an orbital rotation matrix.
    """
    # Check if the tensor has the correct symmetry
    if not tu.check_symmetry(tbt):
      raise ValueError("The tensor does not have the correct symmetry.")
    
    # Perform first factorization of the tensor
    SF_es, SF_vs = SF_terms(tbt)
    SF_sort_idx = np.argsort(np.abs(SF_es))[::-1]
    SF_es = SF_es[SF_sort_idx]
    SF_vs = SF_vs[:, SF_sort_idx]

    nfrags = len(SF_es)
    nmodes = tbt.shape[0]
    nmodals = tbt.shape[1]
    
    # Tensors to store the outputs
    DF_e_ten = np.zeros((nfrags, nmodes, nmodals))
    orbrot_ten = np.zeros((nfrags, nmodes, nmodals, nmodals))
    
    # Perform second factorization of the tensor
    for i in range (nfrags):
       SF_v = SF_vs[:, [i]]                         #Vector storing the information of i'th fragment
       for j in range(nmodes):
          DF_terms_mat = SF_v[j * nmodals : (j + 1) * nmodals, [0]]      #Extract the information of j'th mode
          DF_terms_mat = np.reshape(DF_terms_mat, (nmodals, nmodals))          #Reshape the vector to a matrix of size (nmodals, nmodals) 
          assert np.sum(np.abs(DF_terms_mat - DF_terms_mat.T)) < 1e-10

          DF_es, DF_vs = np.linalg.eigh(DF_terms_mat)          #Second factorization (gives coefficients of diagonal operators and the orbital rotation matrix)
          DF_sort_idx = np.argsort(np.abs(DF_es))[::-1]        #Sort the eigenvalues according to their magnitude
          DF_es = DF_es[DF_sort_idx]                        
          DF_vs = DF_vs[:, DF_sort_idx]                    
          DF_e_ten[i, j, :] = DF_es                            #Store the eigenvalues in the tensor
          orbrot_ten[i, j, :, :] = DF_vs                       #Store the eigenvectors = orbital rotation matrix in the tensor
    return SF_es, DF_e_ten, orbrot_ten
