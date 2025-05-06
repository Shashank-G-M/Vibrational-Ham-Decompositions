
import numpy as np


def transpose_nbt(nbt):
    """
    Permutes axes of a n body tensor from (i, j, ..., p, q, r, s, ...) to (i, p, q, j, r, s, ...). i, j index modes and p, q, r, s index modals.
    This permutation is necessary to reshape the tensor into a matrix that can be decomposed into fragments.
    Currently implemented only for one, two and three body tensors.

    Parameters
    ----------
    nbt : np.ndarray
        An n body tensor of shape (i, j, ..., p, q, r, s, ...).

    Returns
    -------
    np.ndarray
        An n body tensor of shape (i, p, q, j, r, s, ...).
    """

    # Get the bodiness of the tensor
    n = len(nbt.shape) // 3

    if n == 1:
      return nbt
    elif n == 2:
      nbt = np.transpose(nbt, (0, 2, 3, 1, 4, 5))
    elif n == 3:
      nbt = np.transpose(nbt, (0, 3, 4, 1, 5, 6, 2, 7, 8))
    else:
      raise ValueError("Currently only implemented for one, two and three body tensors.")

    return nbt




def check_symmetry(nbt):
    """
    Checks if the n body Hamiltonian tensor has n!*2^n symmetry. For example, obt has 1 symmetry, tbt has 8 symmetries, rbt (three body tensor) has 48 symmetries etc.
    (Currently only implemented for obt and tbt)

    Parameters
    ----------
    nbt : np.ndarray
        An n body tensor of shape (i, p, q, j, r, s, k, t, u, ...). General structure is, mode indices are followed by modal indices of that mode.

    Returns
    -------
    bool
        True if the tensor has correct symmetry, False otherwise.
    """
    # Get the bodiness of the tensor
    n = len(nbt.shape) // 3
    
    issymmetric = 0
    if n == 1:
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (0, 2, 1))))
    
    elif n == 2:
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (0, 2, 1, 3, 4, 5))))
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (0, 1, 2, 3, 5, 4))))
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (0, 2, 1, 3, 5, 4))))
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (3, 1, 2, 0, 4, 5))))
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (3, 2, 1, 0, 4, 5))))
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (3, 1, 2, 0, 5, 4))))
      issymmetric += int(not np.allclose(nbt, np.transpose(nbt, (3, 2, 1, 0, 5, 4))))

    return issymmetric == 0



