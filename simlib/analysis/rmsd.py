"""
rmsd.py
written in Python3
author: C. Lockhart

>>> from simlib import read_pdb, rmsd
>>> trajectory = read_pdb('trajectory1.pdb')
>>> first_structure = trajectory[0]
>>> rmsd = rmsd(trajectory, first_structure)

"""

import numpy as np


# Compute the RMSD between 2 trajectories
def rmsd(q, r=None, pairwise=True, fit=False):
    if r is None:
        r = q
    return rmsd_pairwise(q, r)


def rmsd_pairwise(q, r):
    nq = len(q)
    nr = len(r)
    result = np.zeros((nq, nr))
    for i in range(nq):
        for j in range(nr):
            result[i, j] = np.sqrt(np.mean((q[i].xyz - r[j].xyz) ** 2))
    return result

