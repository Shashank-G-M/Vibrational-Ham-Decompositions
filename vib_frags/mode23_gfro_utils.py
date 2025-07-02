#_______________________________________________________________________________For 2 & 3 mode frag___________________________________________________________________________

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:14:03 2024

@author: shreyas
"""

''' Functions for Christiansen GFRO '''

import numpy as np
import pickle
from numpy.random import uniform
from scipy.linalg import expm
from scipy.optimize import minimize

path_tbt  = ['einsum_path', (1, 3), (1, 2), (0, 1), (0, 1)]
path_thbt = ['einsum_path', (1, 4), (0, 5), (0, 2), (0, 1), (0, 1), (0, 1)]

def num_params(Nl, L):
    '''
    
    Parameters
    ----------
    Nl: # of modals per mode
    L:  # of modes
    
    Returns
    -------
    c:      # of different lambda^{l,m}_{i_l,j_m} parameters
    d:      # of different Gamma^{l,m,n}_{i_l,j_m,k_n} parameters
    u:      # of different theta^{l}_{p_l,q_l} parameters
    c+u:    Total # of parameters
    '''
    c = Nl**2 * L * (L-1) // 2
    d = Nl**3 * L * (L-1) * (L-2) // 6
    u = L * Nl * (Nl - 1) // 2

    return c, d , u, c + d + u

def construct_cartan_tensor(lam, gamma, Nl, L):
    '''

    Parameters
    ----------
    lam : array of lambda parameters
    gamma : array of gamma parameters
    Nl : # of modals per mode
    L : # of modes

    Returns
    -------
    tbt : construct lambda tensor lambda^{i,j}_{pi,qj}
    Only non-zero for i>j
    
    thbt : construct Gamma tensor Gamma^{i,j,k}_{pi,qj,rk}
    Only non-zero for i>j>k
    '''

    tbt  = np.zeros([L,L,Nl,Nl,Nl,Nl])
    thbt = np.zeros([L,L,L,Nl,Nl,Nl,Nl,Nl,Nl])

    tally = 0
    for i in range(L):
        for j in range(i):
            for p in range(Nl):
                for q in range(Nl):
                    # j >= i terms will be zero.
                    # switch i,j & switch p,q to get same tensor ( not here)
                    tbt[i,j,p,q,p,q] += lam[tally]
                    #tbt[j,i,q,p,q,p] += lam[tally]
                    tally            += 1
    
    tally = 0
    for i in range(L):
        for j in range(i):
            for k in range(j):
                for p in range(Nl):
                    for q in range(Nl):
                        for r in range(Nl):
                            # Terms where i>j>k is not true will be zero
                            thbt[i,j,k,p,q,r,p,q,r] += gamma[tally]
                            tally                   += 1
                            
                       
    return tbt, thbt

def construct_orthogonal(theta, Nl, L):
    '''
    Parameters
    ----------
    theta : array of theta parameters
    Nl : # of modals per mode
    L : # of modes

    Returns
    -------
    rotation matrix U^{i},{pi,qi} from theta parameters
    '''

    
    X = np.zeros([L, Nl, Nl])

    tally = 0
    for i in range(L):
        for p in range(Nl):
            for q in range(p+1, Nl):
                X[i,p,q] += theta[tally]
                X[i,q,p] -= theta[tally]
                tally    += 1

    return np.array([expm(X[i,:,:]) for i in range(L)])

def get_fragment(x, Nl, L):
    '''
    Parameters
    ----------
    x : array of fragment parameters
    Nl : # of modals per mode
    L : # of modes

    Returns
    -------
    Two- and three- body tensors of the fragment in the original basis:
    W^{l,m}_{pl,rm,ql,sm} and V^{l,m,n}_{pl,rm,tn,ql,sm,un}
    '''
    c, d, u, p = num_params(Nl, L)
    
    lam    = x[ : c]
    gamma  = x[c : c+d]
    theta  = x[c+d : ]
    #print('# of parameters in lambda', lam.size, '# of parameters in theta',theta.size)

    tbt, thbt = construct_cartan_tensor(lam, gamma, Nl, L)
    O   = construct_orthogonal(theta, Nl, L)

    rot_tbt  = np.einsum('lmpqpq,lpa,mqb,lpc,mqd->lmabcd', tbt, O, O, O, O,optimize=path_tbt)
    rot_thbt = np.einsum('lmnpqrpqr,lpa,mqb,nrc,lpd,mqe,nrf->lmnabcdef', thbt, O, O, O, O, O, O,optimize=path_thbt)
    
    return rot_tbt, rot_thbt

def evaluate_cost_function(x, target_tbt, target_thbt, Nl, L):
    '''
    Parameters
    ----------
    x : array of fragment parameters
    target_tbt : two-body tensor targeted in the optimization
    target_thbt: Target three-body tensor
    Nl : # of modals per mode
    L : # of modes

    Returns
    -------
    The GFRO cost function: 
    sum_{l,m,pl,rm,ql,sm} (g^{l,m}_{pl,rm,ql,sm} - W^{l,m}_{pl,rm,ql,sm})^2 +
    sum_{l,m,n,pl,rm,tn,ql,sm,un} (f^{l,m,n}_{pl,rm,tn,ql,sm,un} - V^{l,m,n}_{pl,rm,tn,ql,sm,un})^2
    '''
    fragment_tbt, fragment_thbt = get_fragment(x, Nl, L)
    diff2         = (fragment_tbt - target_tbt)
    diff3         = (fragment_thbt - target_thbt)

    return np.sum(diff3 * diff3) + np.sum(diff2 * diff2) 

def obtain_gfro_fragment(target_tbt, target_thbt):
    '''
    Parameters
    ----------
    targer_tbt: Target two-body tensor
    target_thbt: Target three-body tensor
        
    Returns
    -------
    Optimized GFRO fragment
    '''
    
    # Number  of modes and modals and parameter counts
    L       = target_tbt.shape[0]
    Nl      = target_tbt.shape[2]
    c, d, u, p = num_params(Nl, L)

    # cost function
    def cost(x):
        return evaluate_cost_function(x, target_tbt, target_thbt, Nl, L)


    # initial guess
    lam_gamma= np.zeros(c+d)
    theta    = np.array([uniform(-np.pi/2, np.pi/2, Nl*(Nl-1)//2) for i in range(L)]).reshape(u)
    x0       = np.concatenate((lam_gamma,theta))
    
    # options
    options = {
        'maxiter' : 10000,
        'disp'    : False,
    }

    #tolerance
    #tol     = 5e-1
    #enum    = L* (L-1) * Nl ** 4
    #fun_tol = (tol / enum) ** 2
    
    def printx(xn):
         with open('Fragments.out', 'a') as f:
             print(f'cost : {cost(xn)}\n', file=f)

    # optimize
    return minimize(cost, x0, method='BFGS', options=options, tol=5e-1, callback=printx)

#Save fragments
def save_params(params, norms):
    '''
    Parameters
    ----------
    params : array of fragment parameters
    norms : array of norm of two-body tensors after removing fragments

    Saves params and norms in a file   
    '''
    filename = "H2S_3MC_fragments.out"
    with open(filename, 'wb') as f:
        pickle.dump([params, norms], f)
    return None

#Load in fragments that were found in earlier calculation
def load_params():
    '''
    Parameters
    ----------
    params : array of fragment parameters
    norms : array of norm of two-body tensors after removing fragments

    Reads params and norms from file   
    '''
    filename = "CO2_fragments.out"
    with open(filename, 'rb') as f:
        params, norms = pickle.load(f)
    return params, norms





