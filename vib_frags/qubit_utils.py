from openfermion import QubitOperator as QO
import scipy as sp
import numpy as np

def sigdag(idx):
    '''
    Get spin creation qubit opeartor at index idx
    
    parameters
    ----------
    idx : int
        Index of the qubit operator.
    
    Returns
    -------
    QubitOperator
        Spin creation qubit operator (X - iY)/2 at index idx.
    '''

    Xidx = QO(f'X{idx}')
    Yidx = QO(f'Y{idx}')
    return (Xidx - 1j*Yidx) / 2


def sig(idx):
    '''
    Get spin annihilation qubit opeartor at index idx
    
    parameters
    ----------
    idx : int
        Index of the qubit operator.
    
    Returns
    -------
    QubitOperator
        Spin annihilation qubit operator (X + iY)/2 at index idx.
    '''

    Xidx = QO(f'X{idx}')
    Yidx = QO(f'Y{idx}')
    return (Xidx + 1j*Yidx) / 2



def Epq_mat(i, p, q, nmodals, nmodes):
    '''
    Get the matrix representation of Epq = sigdag(p) * sig(q) in the subspace of one excitation per mode.
    
    parameters
    ----------
    i : int
        Index of the mode.
    p : int
        Index of creation operator.
    q : int
        Index of annihilation operator.
    nmodals : int
        Number of modals.
    nmodes : int
        Number of modes.
    
    Returns
    -------
    sp.sparse.csc_matrix
        One body operator Ep_iq_i = sigdag(p_i)sig(q_i) as a scipy sparse matrix of shape (nmodals^nmodes, nmodals^nmodes).
    '''
    if p >= nmodals or q >= nmodals:
        raise ValueError("p and q must be less than nmodals")
    if i >= nmodes:
        raise ValueError("i must be less than nmodes")
    
    Epq = sp.sparse.csc_matrix(([1], ([p], [q])), shape=(nmodals, nmodals), dtype=int)
    I_left = sp.sparse.eye(nmodals**i, format='csc', dtype=int)
    I_right = sp.sparse.eye(nmodals**(nmodes-i-1), format='csc', dtype=int)
    
    Epq_full = sp.sparse.kron(I_left, Epq, format='csc')
    Epq_full = sp.sparse.kron(Epq_full, I_right, format='csc')
    
    return Epq_full







def Zp_mat(i, p, nmodals, nmodes):
    '''
    Get the matrix representation of reflection operator Z_pi = 1 - 2(sigdag(pi) * sig(pi)) in the subspace of one excitation per mode.
    
    parameters
    ----------
    i : int
        Mode index
    p : int
        Modal index
    nmodals : int
        Number of modals.
    nmodes : int
        Number of modes.
    
    Returns
    -------
    sp.sparse.csc_matrix
        One body operator Z_pi as a scipy sparse matrix of shape (nmodals^nmodes, nmodals^nmodes).
    '''
    if p >= nmodals:
        raise ValueError("p must be less than nmodals")
    if i >= nmodes:
        raise ValueError("i must be less than nmodes")
    
    data = np.ones(nmodals)
    data[p] = -1
    Zp = sp.sparse.diags(data, offsets=0, format='csc', dtype=int)
    I_left = sp.sparse.eye(nmodals**i, format='csc', dtype=int)
    I_right = sp.sparse.eye(nmodals**(nmodes-i-1), format='csc', dtype=int)
    
    Zp_full = sp.sparse.kron(I_left, Zp, format='csc')
    Zp_full = sp.sparse.kron(Zp_full, I_right, format='csc')
    
    return Zp_full





