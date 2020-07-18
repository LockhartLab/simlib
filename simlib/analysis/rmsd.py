"""
rmsd.py
written in Python3
author: C. Lockhart

>>> from simlib import read_pdb, rmsd
>>> trajectory = read_pdb('trajectory1.pdb')
>>> first_structure = trajectory[0]
>>> rmsd = rmsd(trajectory, first_structure)

"""

from simlib.transform import fit as _fit
import numpy as np


# Compute the RMSD between 2 trajectories
def rmsd(a, b=None, paired=False, fit=True):
    """
    Compute the RMSD.

    If only `a` is provided, then the RMSD is computed between all structures in the Trajectory.
    Otherwise, if `a` and `b` are provided, we compute the RMSD between all structures in `a` and all structures in `b`.


    Parameters
    ----------
    a : Trajectory
    b : Trajectory
    paired : bool
        Are a and b paired? That is, should we compute RMSD between (a[0], b[0]), (a[1], b[1]), etc.? (Default: False)
    fit : bool
        Should structures be fit before RMSD is computed? (Default: True)

    Returns
    -------

    """

    # If a is None, then select b
    if a is None:
        b = a

    # Compute paired?
    if len(a) == len(b) and paired:
        result = _rmsd_paired(a, b, fit=fit)

    # Otherwise, compute RMSD taking a x b
    else:
        result = _rmsd(a, b, fit=fit)

    # Return
    return result


def _rmsd(a, b, fit=True):
    """

    Parameters
    ----------
    a
    b

    Returns
    -------

    """

    # Get the number of elements in a and b
    n_a = len(a)
    n_b = len(b)

    a_xyz = a.xyz
    b_xyz = b.xyz

    # Should we fit the two structures?
    if fit:
        b_xyz = _fit(a, b)

    result = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            result[i, j] = np.sqrt(np.mean((a_xyz[i, :, :] - b_xyz[j, :, :]) ** 2))

    return result
