"""
pdb.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from simlib.analysis.rmsd import rmsd
# noinspection PyProtectedMember
from simlib.io.read import _read_pdb

import numpy as np
from urllib.request import urlopen


# noinspection PyPep8Naming,SpellCheckingInspection
def RCSB(pdb_id, most_common=False):
    """
    Download PDB from RCSB.

    Parameters
    ----------
    pdb_id : str
        Reference ID for the PDB structure.
    most_common : bool
        Extract only the PDB structure with the lowest internal RMSD.

    Returns
    -------
    simlib.Trajectory
        Instance of Trajectory object.
    """

    # Read RCSB and return simlib Trajectory
    result = _read_pdb(urlopen('http://files.rcsb.org/download/' + pdb_id + '.pdb').read().decode('utf-8'))

    # If most_common, compute the pairwise RMSD and select structure structure with minimum average value
    if most_common:
        r = rmsd(result, pairwise=True)
        i = np.argmin(r.mean(axis=0))
        result = result[i]

    # Return
    return result


# print(pdb._data[0])
# pdb.write('1BA4.pdb')

# Take out lines only with CRYST, MODEL, or ATOM
# temp1 = re.sub(r'^(?!(ATOM|CRYST|MODEL)).*$', '', pdb._raw, flags=re.MULTILINE).strip()

# Remove unnecessary whitespace
# temp2 = re.sub(r'\s*$', '', temp1, flags=re.MULTILINE)

# print(temp2)

