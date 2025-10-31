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
    
    comm_idx = nmodals*np.arange(nmodals**i)
    r = nmodals**(nmodes-i-1)                       #dimension of identity matrix to the right of Epiqi
    base = np.arange(r)

    p_offsets = (comm_idx + p)*r
    rows = (base[None, :] + p_offsets[:, None]).ravel()
    q_offsets = (comm_idx + q)*r
    cols = (base[None, :] + q_offsets[:, None]).ravel()
    data = np.ones(len(cols))

    Epq_full = sp.sparse.coo_matrix((data, (rows, cols)), shape = (nmodals**nmodes, nmodals**nmodes), dtype=int).tocsc()    

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
    
    r = nmodals**(nmodes-i-1)
    data = np.repeat(data, r)
    l = nmodals**i
    data = np.tile(data, l)
    Zp_full = sp.sparse.diags(data, format = 'csc', dtype=int)
    
    return Zp_full




def get_QOP_of_Pauli_LCU(C = None, obt = None, tbt = None, trbt = None, cutoff = 1e-4, opt = False):
    '''
    Get the QubitOperator corresponding to sub-optimal Pauli LCU of the vibrational Hamiltonian defined by the given tensor/s
    
    parameters
    ----------
    C : int (optional)
        Constant term
    obt: np.array (optional)
      one-body tensor of shape (nmodes, nmodals, nmodals).
      If opt = True, this represents the modified one body tensor and not the original Hamiltonian tensor.
    tbt: np.array (optional)
      two-body tensor of shape (nmodes, nmodals, nmodals,nmodes, nmodals, nmodals)  
      If opt = True, this represents the modified two body tensor and not the original Hamiltonian tensor.
    trbt: np.array (optional)
      three-body tensor of shape (nmodes, nmodals, nmodals,nmodes, nmodals, nmodals,nmodes, nmodals, nmodals)  
      If opt = True, this represents the modified three body tensor and not the original Hamiltonian tensor.
    cutoff: float (optional)
      cutoff to prune the tensor elements. Defaults to 1e-4 (appropriate for cm-1 units)
    opt: bool (optional)
      if True, returns QubitOperator corresponding to optimal Pauli LCU.
    
    Returns
    -------
    openfermion.QubitOperator
        QubitOperator corresponding to the constant term
    openfermion.QubitOperator
        QubitOperator corresponding to the obt
    openfermion.QubitOperator
        QubitOperator corresponding to the tbt
    openfermion.QubitOperator
        QubitOperator corresponding to the trbt
    '''

    #Initialize the opeartors
    C_op = QO.zero()
    obt_op = QO.zero()
    tbt_op = QO.zero()
    trbt_op = QO.zero()

    if C is not None:
        C_op = C_op + C*QO.identity()

    nmodes = next(ten.shape[0] for ten in (obt, tbt, trbt) if ten is not None)
    nmodals = next(ten.shape[-1] for ten in (obt, tbt, trbt) if ten is not None)

    #Define the qubit opeartor Q_op that appears in the sub-optimal Pauli LCU.
    if opt == False:
        def Q_op(i,p,q,nmodals):
            idx_p = i*nmodals + p
            idx_q = i*nmodals + q
            if p > q:
                return QO(f'X{idx_p}')*QO(f'X{idx_q}')
            if p < q:
                return QO(f'Y{idx_p}')*QO(f'Y{idx_q}')
            else:
                return QO.identity() - QO(f'Z{idx_p}')
    else:
        def Q_op(i,p,q,nmodals):
            idx_p = i*nmodals + p
            idx_q = i*nmodals + q
            if p > q:
                return QO(f'X{idx_p}')*QO(f'X{idx_q}')
            if p < q:
                return QO(f'Y{idx_p}')*QO(f'Y{idx_q}')
            else:
                return -QO(f'Z{idx_p}')

            

    # Precompute Q_ops[i,p,q] as dict or nested list for fast lookup
    Q_ops = {}
    for i in range(nmodes):
        for p in range(nmodals):
            for q in range(nmodals):
                Q_ops[(i,p,q)] = Q_op(i, p, q, nmodals)


    # Prepare caches for pairwise products to avoid recomputing
    pair_cache = {}   # key = ((i,p,q),(j,r,s)) -> Q_ops[(i,p,q)] * Q_ops[(j,r,s)]

    # Helper to iterate only over nonzero entries of a coefficient tensor
    def nonzero_index_iter(tensor):
        # handles numpy arrays or None
        if tensor is None:
            return
        # numpy: use argwhere to iterate indices where tensor != 0
        # use .flat to skip zero floats; if tensor is symbolic coefficients, adjust accordingly
        for idx in np.argwhere(np.abs(tensor) > cutoff):
            yield tuple(int(i) for i in idx)  # idx is ndarray of ints

    # Build index lists for tbt and trbt to reduce nested loops
    # For tbt: indices are (i,p,q,j,r,s)
    tbt_entries = []
    if tbt is not None:
        for idx in nonzero_index_iter(tbt):
            tbt_entries.append(idx)  # tuple length 6

    # For trbt: indices are (i,p,q,j,r,s,k,t,u)
    trbt_entries = []
    if trbt is not None:
        for idx in nonzero_index_iter(trbt):
            trbt_entries.append(idx)  # tuple length 9

    # For obt: we can just iterate nonzero (i,p,q)
    if obt is not None:
        for (i,p,q) in nonzero_index_iter(obt):
            coeff = obt[i,p,q]
            if coeff:
                obt_op += coeff * Q_ops[(i,p,q)]

    # For tbt: iterate nonzero entries; use pair cache
    if tbt is not None:
        for (i,p,q,j,r,s) in tbt_entries:
            coeff = tbt[i,p,q,j,r,s]
            if not coeff:
                continue
            key = ((i,p,q),(j,r,s))
            prod = pair_cache.get(key)
            if prod is None:
                prod = Q_ops[(i,p,q)] * Q_ops[(j,r,s)]
                pair_cache[key] = prod
            tbt_op += coeff * prod

    # For trbt: iterate nonzero entries; reuse pair_cache for first two
    if trbt is not None:
        for (i,p,q,j,r,s,k,t,u) in trbt_entries:
            coeff = trbt[i,p,q,j,r,s,k,t,u]
            if not coeff:
                continue
            key_ij = ((i,p,q),(j,r,s))
            pair_ij = pair_cache.get(key_ij)
            if pair_ij is None:
                pair_ij = Q_ops[(i,p,q)] * Q_ops[(j,r,s)]
                pair_cache[key_ij] = pair_ij
            # multiply the third operator
            triple = pair_ij * Q_ops[(k,t,u)]
            trbt_op += coeff * triple

    return C_op, obt_op, tbt_op, trbt_op
