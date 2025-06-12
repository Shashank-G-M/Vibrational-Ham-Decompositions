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
