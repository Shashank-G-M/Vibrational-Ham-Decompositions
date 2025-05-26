#Utility file to perform triple factorization (tf) of three mode vibrational Hamiltonian in Christiansen form
#The idea is similar to MPF. We first reshape the tensor into a rectangular matrix and perform SVD. 
#The right singular vectors are then reshaped into a tensor which have the same symmetries as the two-mode tensor.
#This allows to perform double factorization of the right singular vectors.

import numpy as np
from . import tensor_utils as tu
from . import df_utils as df
from opt_einsum import contract
from numpy.linalg import svd






def TF_terms(trbt, cutoff=1e-10, force_sym = False):
  """
  Computes triple factorization of a three body tensor. Returns six objects that are needed to reconstruct a TF fragment.
  Error will be raised if the input trbt does not have 48 fold symmetry.

  Parameters
  ----------
  trbt : np.ndarray
      A three body tensor of shape (i, p, q, j, r, s, k, t, u).
  cutoff : float
      cutoff for truncation of fragments. Default is 1e-10.
  force_sym : bool
      If True, all relevant symmetries are enforced in the intermediate tensors. This is done to avoid numerical instabilities.
      Default is True.

  Returns
  -------
  np.ndarray(nfrags,)
      Singular values from first factorization sorted according to their magnitude.
  np.ndarray(nfrags, nmodes, nmodals)
      Eigenvalues from factorization of the one-body tensor (left singular vector) of each fragment sorted according to their magnitude witihn each mode.
      The first axis denotes the fragment index.
      The second axis denotes the mode index. 
  np.ndarray(nfrags, nmodes, nmodals, nmodals)
      Orbital rotation matrix for diagonalizing the one-body operator on each mode of each fragment.
      The first axis denotes the fragment index.
      The second axis denotes the mode index.
      The third and fourth axis denote the element of an orbital rotation matrix.
  list [np.ndarray(nsubfrags, )]
      Each element of the list is an ndarray that stores eigenvalues from factorization of the two-body tensor (right singular vector) of each fragment, 
      sorted according to their magnitude. 
  list [np.ndarray(nsubfrags, nmodes, nmodals)]
      Each element of the list is an ndarray that stores eigenvalues from factorization of eigenvectors of two-body tensor.
      The first axis denotes the subfragment index from factorizing the two-body tensor.
      The second axis denotes the mode index.
  list [np.ndarray(nsubfrags, nmodes, nmodals, nmodals)]
      Each element of the list is an ndarray that stores orbital rotation matrix for diagonalizing the two-body operator on each mode of each subfragment.
      The first axis denotes the subfragment index from factorizing the two-body tensor.
      The second axis denotes the mode index.
      The third and fifth axis denote the element of an orbital rotation matrix.
  """


  # Check if the tensor has the correct symmetry
  if not tu.check_symmetry(trbt):
    raise ValueError("The tensor does not have the correct symmetry.")
  
  # Perform first factorization of the tensor
  trbt_mat = np.reshape(trbt, (trbt.shape[0] * trbt.shape[1] * trbt.shape[2], trbt.shape[3] * trbt.shape[4] * trbt.shape[5] * trbt.shape[6] * trbt.shape[7] * trbt.shape[8]))
  left_svs, sing_vals, right_svsT = svd(trbt_mat, full_matrices=False) #Perform SVD on the reshaped tensor
  right_svs = right_svsT.T                                            

  #Truncate fragments
  lrg_frag_idx = np.where(np.abs(sing_vals) > cutoff)[0]              #Get the indices of the fragments with eigenvalues greater than the cutoff
  sing_vals = sing_vals[lrg_frag_idx]                            
  left_svs = left_svs[:, lrg_frag_idx]
  right_svs = right_svs[:, lrg_frag_idx]

  #Sort the singular values and vectors
  sort_idx = np.argsort(np.abs(sing_vals))[::-1]                    #Sort the singular values according to their magnitude
  sing_vals = sing_vals[sort_idx]
  left_svs = left_svs[:, sort_idx]
  right_svs = right_svs[:, sort_idx]

  nfrags = len(sing_vals)
  nmodes = trbt.shape[0]
  nmodals = trbt.shape[-1]
  
  output1 = sing_vals
  output2 = np.zeros((nfrags, nmodes, nmodals))
  output3 = np.zeros((nfrags, nmodes, nmodals, nmodals))
  output4 = []
  output5 = []
  output6 = []

  print ("SVD complete")

  for a in range(nfrags):
    u = left_svs[:, [a]]                                
    u = np.reshape(u, (nmodes, nmodals, nmodals))
    v = right_svs[:, [a]]
    v = np.reshape(v, (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))

    
    #Diagonzalize the one-body operator
    for i in range (nmodes):
      ui_mat = u[i, :, :]          #Extract the information of i'th mode
      
      #Check symmetry
      try:  
        diff = np.linalg.norm(ui_mat - ui_mat.T, 1)
        assert diff < 1e-6
      except AssertionError:
        print ("frag index = ", a, "Mode index = ", i, "Diff = ", diff)
        if force_sym:
          ui_mat = (ui_mat + ui_mat.T)/2
        raise Warning("The matrix is not symmetric.")
      
      Chi_i, Bi = np.linalg.eigh(ui_mat)          #Diagonalize the one-body operator (gives coefficients of diagonal operators and the orbital rotation matrix)
      Chi_sort_idx = np.argsort(np.abs(Chi_i))[::-1]        #Sort the eigenvalues according to their magnitude
      Chi_i = Chi_i[Chi_sort_idx]
      Bi = Bi[:, Chi_sort_idx]

      output2[a, i, :] = Chi_i                            #Store the eigenvalues in the tensor
      output3[a, i, :, :] = Bi                            #Store the eigenvectors = orbital rotation matrix in the tensor

    #Check tbt symmetry
    print (tu.check_symmetry(v))
    if force_sym:
      v = tu.symmetrize_tbt(v, force_sym = True)

    #Factorizing the two-body operator
    output4_a, output5_a, output6_a = df.DF_terms(v, cutoff=cutoff, force_sym = force_sym)

    #Store the results in a list
    output4.append(output4_a)
    output5.append(output5_a)
    output6.append(output6_a)
  
  return output1, output2, output3, output4, output5, output6


  






