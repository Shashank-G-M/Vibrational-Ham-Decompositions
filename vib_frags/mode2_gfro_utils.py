#_______________________________________________________________________________For 2 mode frag___________________________________________________________________________

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

path = ['einsum_path', (1, 3), (1, 2), (0, 1), (0, 1)]

def num_params(Nl,L):
    '''
    
    Parameters
    ----------
    Nl: # of modals per mode
    L:  # of modes
    
    Returns
    -------
    c:      # of different lambda^{l,m}_{i_l,j_m} parameters
    u:      # of different theta^{l}_{p_l,q_l} parameters
    c+u:    Total # of parameters
    '''
    c = Nl**2 * L * (L-1) // 2
    u = L * Nl * (Nl - 1) // 2

    return c, u, c + u

def construct_cartan_tensor(lam, Nl, L):
    '''

    Parameters
    ----------
    lam : array of lambda parameters
    Nl : # of modals per mode
    L : # of modes

    Returns
    -------
    tbt : construct lambda tensor lambda^{i,j}_{pi,qj}
    '''

    tbt = np.zeros([L,L,Nl,Nl,Nl,Nl])

    tally = 0
    for i in range(L):
        for j in range(i):
            for p in range(Nl):
                for q in range(Nl):
                    # j>i terms will be zero.
                    # switch i,j & switch p,q to get same tensor
                    tbt[i,j,p,q,p,q] += lam[tally]
                    #tbt[j,i,q,p,q,p] += lam[tally]
                    tally            += 1

    return tbt

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
    Two-body tensor of the fragment in the original basis:
    W^{l,m}_{pl,rm,ql,sm} 
    '''
    c, u, p = num_params(Nl, L)

    
    lam    = x[ : c]
    theta  = x[c : ]
    #print('# of parameters in lambda', lam.size, '# of parameters in theta',theta.size)

    tbt = construct_cartan_tensor(lam, Nl, L)
    O   = construct_orthogonal(theta, Nl, L)

    return np.einsum('lmpqpq,lpa,mqb,lpc,mqd->lmabcd', tbt, O, O, O, O,optimize=path)
   

def evaluate_cost_function(x, target_tbt, Nl, L):
    '''
    Parameters
    ----------
    x : array of fragment parameters
    target_tbt : two-body tensor targeted in the optimization
    Nl : # of modals per mode
    L : # of modes

    Returns
    -------
    The GFRO cost function: 
    sum_{l,m,pl,rm,ql,sm} (g^{l,m}_{pl,rm,ql,sm} - W^{l,m}_{pl,rm,ql,sm})^2
    '''
    fragment_tbt = get_fragment(x, Nl, L)
    diff         = (fragment_tbt - target_tbt)

    return np.sum(diff * diff)

def obtain_gfro_fragment(target_tbt):
    '''
    Returns
    -------
    Optimized GFRO fragment
    '''
    
    # Number  of modes and modals and parameter counts
    L       = target_tbt.shape[0]
    Nl      = target_tbt.shape[2]
    c, u, p = num_params(Nl, L)

    # cost function
    def cost(x):
        return evaluate_cost_function(x, target_tbt, Nl, L)

    # gradient function
    def grad(x):
        return evaluate_gradient_function(x, target_tbt, Nl, L)

    # initial guess
    lam      = np.zeros(c)
    theta    = np.array([uniform(-np.pi/2, np.pi/2, Nl*(Nl-1)//2) for i in range(L)]).reshape(u)
    x0       = np.concatenate((lam,theta))
    
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
    return minimize(cost, x0, method='BFGS', options=options, tol=1e-7, callback=printx, jac = grad)

#Save fragments
def save_params(params, norms):
    '''
    Parameters
    ----------
    params : array of fragment parameters
    norms : array of norm of two-body tensors after removing fragments

    Saves params and norms in a file   
    '''
    filename = "CO2_fragments.out"
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

#
#    implementation of the gradient
#

def fragment_cartan_coef_derivative(O, l, m, i, j):
    '''
    returns tensor
    a[pl,rm,ql,sm] = O^{l}_{il,pl} O^{l}_{il,ql} O^{m}_{jm,rm} O^{m}_{jm,sm}
    '''
    return np.einsum('p,r,q,s->prqs', O[l,i], O[m,j], O[l,i], O[m,j])

def fragment_orbital_rotation_derivative(l, O, lams, deltaL, deltaNl):
    L  = O.shape[0]
    Nl = O.shape[1]

    tbt = np.zeros([L,L,Nl,Nl,Nl,Nl,Nl,Nl])
    tbt += np.einsum('x,up,tq,yjr,yjs,ytj->xyprqstu', deltaL[l], deltaNl, O[l], O, O, lams[l,:,:,:])
    tbt += np.einsum('x,uq,tp,yjr,yjs,ytj->xyprqstu', deltaL[l], deltaNl, O[l], O, O, lams[l,:,:,:])
    tbt += np.einsum('y,ur,ts,xip,xiq,xit->xyprqstu', deltaL[:,l], deltaNl, O[l], O, O, lams[:,l,:,:])
    tbt += np.einsum('y,us,tr,xip,xiq,xit->xyprqstu', deltaL[:,l], deltaNl, O[l], O, O, lams[:,l,:,:])

    return tbt

def get_fragment_orbital_rotation_derivative(lam, O, Nl, L):
    '''
    Parameters
    ----------
    lam : array of fragment parameters lamda
    O: rotation matrices O^{l}_{pl,ql}
    Nl : # of modals per mode
    L : # of modes  
    
    Returns
    -------
    The derivative dW/dU  
    '''
    deltaNl = np.identity(Nl)
    deltaL  = np.identity(L)

    lams_matrix = np.zeros([L, L, Nl, Nl])

    tally = 0
    for l in range(L):
        for m in range(l+1, L):
            for p in range(Nl):
                for q in range(Nl):
                    lams_matrix[l,m,p,q] += lam[tally]
                    lams_matrix[m,l,q,p] += lam[tally]
                    tally                += 1
    
    return np.array([fragment_orbital_rotation_derivative(l, O, lams_matrix, deltaL, deltaNl) for l in range(L)])

def construct_antisymmetric(angles, N):

    X = np.zeros([N,N])

    tally = 0
    for p in range(N):
        for q in range(p+1, N):
            X[p,q] += angles[tally]
            X[q,p] -= angles[tally]
            tally  += 1

    return X

def get_kappa(i, N):

    kappa = np.zeros([N,N])

    tally = 0
    for p in range(N):
        for q in range(p+1, N):
            if tally == i:
                kappa[p,q] += 1
                kappa[q,p] -= 1
            tally += 1

    return kappa

def orbital_rotation_angle_derivative(angles, i, N):
    kappa = get_kappa(i, N)
    X     = construct_antisymmetric(angles, N)
    D, P  = np.linalg.eig(X)

    expD = np.zeros([N,N], dtype=np.complex128)
    for a, lam in enumerate(D):
        expD[a,a] += np.exp(lam)

    I = P.conj().T @ kappa @ P
    for a in range(N):
        for b in range(N):
            if np.abs(D[a] - D[b]) > 1e-8:
                I[a,b] *= (np.exp(D[a] - D[b]) - 1) / (D[a] - D[b])

    return np.real(P @ I @ expD @ P.conj().T)

def compute_coef_derivatives(x, O, dCdW):
    '''
    Parameters
    ----------
    x : array of fragment parameters
    O: rotation matrices O^{l}_{pl,ql}
    dCdW : Derivative of cost function wrt fragment tensor
    
    Returns
    -------
    The derivative of the cost function wrt lambda coefficients 
    '''
    L  = O.shape[0]
    Nl = O.shape[1]

    c, u, p = num_params(Nl, L)

    output = np.zeros(c)
    tally = 0
    for l in range(L):
        for m in range(l+1, L):
            for i in range(Nl):
                for j in range(Nl):
                    dW_lmdlam_lm   = fragment_cartan_coef_derivative(O, l, m, i, j)
                    dW_mldlam_lm   = fragment_cartan_coef_derivative(O, m, l, j, i)
                    output[tally] += np.einsum('prqs,prqs->', dCdW[l,m], dW_lmdlam_lm) + np.einsum('prqs,prqs->', dCdW[m,l], dW_mldlam_lm)
                    tally         += 1
    
    return output

def compute_angle_derivatives(x, O, dCdW):
    '''
    Parameters
    ----------
    x : array of fragment parameters
    O: rotation matrices O^{l}_{pl,ql}
    dCdW : Derivative of cost function wrt fragment tensor
    
    Returns
    -------
    The derivative of the cost function wrt theta angles 
    '''
    L  = O.shape[0]
    Nl = O.shape[1]

    c, u, p = num_params(Nl, L)
    t = int(Nl*(Nl-1)/2)
    lam    = x[ : c]
    theta  = x[c : ]
    angles = np.zeros([L, t])
    tally1  = 0
    
    for l in range(L):
        tally2 = 0
        for p in range(Nl):
            for q in range(p+1, Nl):
                angles[l,tally2] = theta[tally1]
                tally1 += 1
                tally2 += 1

    dWdO = get_fragment_orbital_rotation_derivative(lam, O, Nl, L)

    output = np.zeros(u)
    tally = 0
    for l in range(L):
        for i in range(t):
            dOdtheta_i = orbital_rotation_angle_derivative(angles[l], i, Nl)
            dWdtheta_i = np.einsum('xyprqstu,tu->xyprqs', dWdO[l], dOdtheta_i)
            output[tally] += np.einsum('xyprqs,xyprqs->', dCdW, dWdtheta_i)
            tally += 1

    return output

def evaluate_gradient_function(x, target_tbt, Nl, L):
    '''
    Parameters
    ----------
    x : array of fragment parameters
    target_tbt : two-body tensor targeted in the optimization
    Nl : # of modals per mode
    L : # of modes

    Returns
    -------
    The gradient of the GFRO cost function
    '''
    c, u, p = num_params(Nl, L)

    lam    = x[ : c]
    theta  = x[c : ]
    
    O            = construct_orthogonal(theta, Nl, L)
    fragment_tbt = get_fragment(x, Nl, L)
    dCdW         = 2 * (fragment_tbt - target_tbt)

    grad       = np.zeros(p)
    grad[ : c] = compute_coef_derivatives(x, O, dCdW)
    grad[c : ] = compute_angle_derivatives(x, O, dCdW)

    return grad

