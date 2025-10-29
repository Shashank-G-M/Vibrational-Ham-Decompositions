
import numpy as np
from . mode2_gfro_utils import get_fragment as get_fragment_2
from . mode23_gfro_utils import get_fragment as get_fragment_23
import pickle

#Load in fragments that were found in earlier calculation
def load_23_mode_params(mol, nmodes, nmodals, loc=True):
    '''
    Parameters
    ----------
    params : array of fragment parameters
    norms : array of norm of two-body tensors after removing fragments

    Reads params and norms from file   
    Return fragments such that residual norm <= 0.05
    '''
    if loc == True:
      with open(rf"VSCF_basis_loc_frags/two_and_three_mode/{mol}_4T3M_VSCFfragments_loc{str(nmodals)}modals.out", 'rb') as f:
          params, norms = pickle.load(f)
    else:
      with open(rf"VSCF_basis_frags/two_and_three_mode/{mol}_4T3M_VSCFfragments_{str(nmodals)}modals.out", 'rb') as f:
          params, norms = pickle.load(f)

 
    nf = len(params)
    H2_frag = np.zeros((nf,nmodes,nmodes,nmodals,nmodals,nmodals,nmodals))
    H3_frag = np.zeros((nf,nmodes,nmodes,nmodes,nmodals,nmodals,nmodals,nmodals,nmodals,nmodals))
    for i in range(nf):
        H2_frag[i], H3_frag[i] = get_fragment_23(params[i], nmodals, nmodes)
        if np.sqrt(norms[i+1]/norms[0]) >= 0.001:
            H2_frag[i], H3_frag[i] = get_fragment_23(params[i], nmodals, nmodes)
        else:
            H2_frag[i], H3_frag[i] = get_fragment_23(params[i], nmodals, nmodes)
            print(f'Total number of fragments included = {i+1}')
            break
    nf = i+1  #Number of fragments included
    H2_frag = H2_frag[:nf] 
    H3_frag = H3_frag[:nf]

    
    return params[:nf], norms[:nf+1], H2_frag, H3_frag

#_______________________________________________________________________________For 2 mode frag___________________________________________________________________________

#Load in fragments that were found in earlier calculation
def load_2_mode_params(mol, nmodes, nmodals, loc=True):
    '''
    Parameters
    ----------
    params : array of fragment parameters
    norms : array of norm of two-body tensors after removing fragments

    Reads params and norms from file   
    Return fragments such that residual norm <= 0.05
    '''
    if loc == True:
      with open(rf"VSCF_basis_loc_frags/only_two_mode/{mol}_VSCFfragments_loc{str(nmodals)}modals.out", 'rb') as f:
          params, norms = pickle.load(f)
    else:
      with open(rf"VSCF_basis_frags/only_two_mode/{mol}_VSCFfragments_{str(nmodals)}modals.out", 'rb') as f:
          params, norms = pickle.load(f)
 
    nf = len(params)
    H2_frag = np.zeros((nf,nmodes,nmodes,nmodals,nmodals,nmodals,nmodals))
    for i in range(nf):
       H2_frag[i] = get_fragment_2(params[i], nmodals, nmodes)
    #   if np.sqrt(norms[i+1]/norms[0]) >= 0.05:
    #      H2_frag[i] = get_fragment_2(params[i], nmodals, nmodes)
    #   else:
    #      H2_frag[i] = get_fragment_2(params[i], nmodals, nmodes)
    #      print(f'Total number of fragments included = {i+1}')
    #      break
    # nf = i+1  #Number of fragments included
    # H2_frag = H2_frag[:nf]
    
    return params[:nf], norms[:nf+1], H2_frag