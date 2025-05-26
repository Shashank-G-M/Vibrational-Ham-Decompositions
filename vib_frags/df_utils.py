#Utility file to perfeorm double factorization of two mode vibrational Hamiltonian in Christiansen form
import numpy as np
from . import tensor_utils as tu
from opt_einsum import contract

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
    
    # Check if the reshaped tensor is symmetric
    assert np.sum(np.abs(tbt_mat - tbt_mat.T)) < 1e-6

    # Compute the eigenvalues and eigenvectors of the reshaped tensor
    eigvals, eigvecs = np.linalg.eigh(tbt_mat)
    
    # Sort the eigenvalues and eigenvectors according to the magnitude of the eigenvalues
    sort_idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    return eigvals, eigvecs





def DF_terms(tbt, cutoff=1e-10, force_sym = False):
    """
    Computes double factorization of a two body tensor. Returns three objects: Eigenvalues from first factorization sorted according to their magnitude, 
    Eigenvalues from second factorization sorted according to their magnitude encoded in a tensor, and the orbital rotation matrix for each node of each 
    fragment encoded in a tensor.
    Error will be raised if the input tbt does not have eight fold symmetry.

    Parameters
    ----------
    tbt : np.ndarray
        A two body tensor of shape (i, p, q, j, r, s).
    cutoff : float
        cutoff for truncation of fragments. Default is 1e-10.
    force_sym : bool
        If True, an intermediate matrix after single factorization is forced to be symmetric. 
        This is done to avoid numerical instabilities in the second eigenvalue decomposition.
        Ideally, this intermediate matrix should be symmetric, but due to numerical errors, it may not be.
        Default is True.

    Returns
    -------
    np.ndarray (nfrags,)
        Eigenvalues from first factorization sorted according to their magnitude.
    np.ndarray (nfrags, nmodes, nmodals)
        Eigenvalues from second factorization for each fragment and mode sorted according to their magnitude.
    np.ndarray (nfrags, nmodes, nmodals, nmodals)
        Orbital rotation matrix for each mode of each fragment.
        The first axis denotes the fragment index.
        The second axis denotes the mode index.
        The third and fourth axis denote the element of an orbital rotation matrix.
    """
    # Check if the tensor has the correct symmetry
    if not tu.check_symmetry(tbt):
      raise ValueError("The tensor does not have the correct symmetry.")
    
    # Perform first factorization of the tensor
    SF_es, SF_vs = SF_terms(tbt)

    #Truncate fragments
    lrg_frag_idx = np.where(np.abs(SF_es) > cutoff)[0]              #Get the indices of the fragments with eigenvalues greater than the cutoff
    SF_es = SF_es[lrg_frag_idx]                            
    SF_vs = SF_vs[:, lrg_frag_idx]

    nfrags = len(SF_es)
    nmodes = tbt.shape[0]
    nmodals = tbt.shape[1]
    
    # Tensors to store the outputs
    DF_e_ten = np.zeros((nfrags, nmodes, nmodals))
    orbrot_ten = np.zeros((nfrags, nmodes, nmodals, nmodals))
    
    # Perform second factorization of the tensor
    for h in range (nfrags):
       SF_v = SF_vs[:, [h]]                                #Vector storing the information of h'th fragment
       SF_v = np.reshape(SF_v, (nmodes, nmodals, nmodals)) #Reshape the vector in to a tensor with each row storing a matrix of size (nmodals, nmodals)
       for i in range(nmodes):
          DF_terms_mat = SF_v[i, :, :]          #Extract the information of i'th mode
          if force_sym:
            DF_terms_mat = (DF_terms_mat + DF_terms_mat.T)/2
          try:
            diff = np.linalg.norm(DF_terms_mat - DF_terms_mat.T, 2)
            assert diff < 1e-6
          except AssertionError:
            print ("frag index = ", h, "Mode index = ", i, "Diff = ", diff)
            raise AssertionError("The matrix is not symmetric.")

          DF_es, DF_vs = np.linalg.eigh(DF_terms_mat)          #Second factorization (gives coefficients of diagonal operators and the orbital rotation matrix)
          DF_sort_idx = np.argsort(np.abs(DF_es))[::-1]        #Sort the eigenvalues according to their magnitude
          DF_es = DF_es[DF_sort_idx]                        
          DF_vs = DF_vs[:, DF_sort_idx]                    
          DF_e_ten[h, i, :] = DF_es                            #Store the eigenvalues in the tensor
          orbrot_ten[h, i, :, :] = DF_vs                       #Store the eigenvectors = orbital rotation matrix in the tensor
    return SF_es, DF_e_ten, orbrot_ten









def get_DF_tbts(tbt, cutoff = 1e-5, force_sym = True):
  """
  Perform double factorization and return a list of two body tensors for each fragment.

  Parameters
  ----------
  tbt : np.ndarray
      A two body tensor of shape (i, p, q, j, r, s).
  cutoff : float
      cutoff for truncation of fragments. Default is 1e-5.
  force_sym : bool
      If True, all relevant symmetries are enforced in the intermediate tensors. This is done to avoid numerical instabilities.
      Default is True.
  
  Returns
  -------
  list of np.ndarray
      A list of two body tensors for each fragment.
  """
  SF_es, DF_e_ten, orbrot_ten = DF_terms(tbt, cutoff, force_sym)
  frag_tens = []                                          #List to store fragment tensors

  print (len(SF_es), " fragments found with eigenvalues greater than ", cutoff)
  for h in range(len(SF_es)):
    coeff_mat = DF_e_ten[h, :, :]                         #Extract the coefficients of the h'th fragment
    u = orbrot_ten[h, :, :, :]                            #Extract the orbital rotation matrix of the h'th fragment for all modes
    frag_ten = contract('ik,jl,ipk,iqk,jrl,jsl -> ipqjrs', coeff_mat, coeff_mat, u, u, u, u)
    frag_ten *= SF_es[h]                                  #Multiply the fragment with the eigenvalues of the first factorization
    frag_tens.append(frag_ten)
  return frag_tens








def get_largest_DFF_energies(SF_es, DF_e_ten):
    """
    Calculate the absolute largest energies of double factorized fragments within the subspace of one excitation per mode.
    The inputs to this function can be obtained from DF_terms function.

    Parameters
    ---------
    SF_es : np.ndarray
        Eigenvalues from first factorization sorted according to their magnitude.
    DF_e_ten : np.ndarray
        Eigenvalues from second factorization for each fragment and node sorted according to their magnitude.
        The first axis denotes the fragment index.
        The second axis denotes the mode index.
        The third axis denotes the eigenvalue index.

    Returns
    -------
    list
        The absolute largest energies of double factorized fragments.
    """
    nmodes = DF_e_ten.shape[1]

    frag_emxs = []
    for h in range (len(SF_es)):
        SF_e = SF_es[h]
        frag_emax1 = np.abs(sum([np.max(DF_e_ten[h, i, :]) for i in range(nmodes)]))
        frag_emax2 = np.abs(sum([np.min(DF_e_ten[h, i, :]) for i in range(nmodes)]))
        frag_emax = np.abs(SF_e*max(frag_emax1, frag_emax2))
        frag_emxs.append(frag_emax)
    return frag_emxs