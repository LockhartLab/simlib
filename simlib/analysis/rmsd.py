"""
rmsd.py
written in Python3
author: C. Lockhart

>>> from simlib import read_pdb, rmsd
>>> trajectory = read_pdb('trajectory1.pdb')
>>> first_structure = trajectory[0]
>>> rmsd = rmsd(trajectory, first_structure)

"""


# Compute the RMSD between 2 trajectories
def rmsd(trajectory1, trajectory2=None, pairwise=False, fit=True):
    pass
