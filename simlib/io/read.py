"""
read.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from simlib.framework import Topology, Trajectory

import numpy as np
# from numpy.lib.recfunctions import drop_fields, structured_to_unstructured
import pandas as pd
import re


# Read PDB
# TODO currently the only backend will by pandas; in future, expand to Cython or C or Fortran backend
def read_pdb(filename, backend='python'):
    """
    Read PDB file and return Trajectory

    PDB format can be described in depth at `<http://www.wwpdb.org/documentation/file-format>`_.

    Parameters
    ----------
    filename : str
        Name of PDB file to be read
    backend : str
        (Default: 'pandas').

    Returns
    -------
    simlib.Trajectory
        Trajectory of PDB
    """

    # Make sure we know we're using the pandas backend
    if backend.lower() != 'python':
        raise AttributeError('only python backend presently supported')

    # Open file, read in all records
    with open(filename, 'r') as buffer:
        records = buffer.read()
        # records = _atom_reader(buffer)

    # Return
    return _read_pdb(records)


def _read_pdb(records):
    # Filter out atom records
    # TODO this will be slow for large PDB files; perhaps move to Cython or C backend
    atoms = re.sub(r'^(?!ATOM).*$', '', records, flags=re.MULTILINE).replace('\n\n', '\n').lstrip()

    # Sections of PDB
    sections = np.array([
        (6, 'record', '<U6'),
        (5, 'atom_id', 'int'),
        (5, 'atom', '<U5'),
        (5, 'residue', '<U5'),
        (1, 'chain', '<U1'),
        (4, 'residue_id', 'int'),
        (4, 'blank', '<U1'),
        (8, 'x', 'float'),
        (8, 'y', 'float'),
        (8, 'z', 'float'),
        (6, 'alpha', 'float'),
        (6, 'beta', 'float'),
        (9, 'segment', '<U9'),
        (2, 'element', '<U2')
    ], dtype=[('width', 'i1'), ('column', '<U10'), ('type', '<U10')])

    # Read in
    data = np.genfromtxt(atoms.split('\n'), delimiter=sections['width'], dtype=sections['type'], autostrip=True)
    # data.dtype.names = sections['column']
    data = pd.DataFrame(data.tolist(), columns=sections['column'])

    # Drop extraneous columns
    # data = drop_fields(data, 'blank')
    data = data.drop(columns='blank')

    # TODO this should also be done for residue_id probably
    # If the PDB starts at atom_id = 1, change to 0-index
    if data['atom_id'].min() == 1:
        data['atom_id'] -= 1

    # Determine number of structures in PDB
    n_structures = np.unique(np.bincount(data['atom_id'].values))
    # n_structures = data.pivot_table(index='atom_id', values='record', aggfunc='count')['record'].unique()
    if len(n_structures) != 1:
        raise AttributeError('inconsistent record counts in PDB')
    n_structures = n_structures[0]

    # Separate out dynamic columns for Trajectory and static Topology data
    dynamical_columns = ['x', 'y', 'z']
    # static_columns = [column for column in data.dtype.names if column not in dynamical_columns]
    static_columns = [column for column in data.columns if column not in dynamical_columns]

    # Create Topology first
    # TODO what happens when alpha and beta differ by structures? Should these be stored in Trajectory?
    data['alpha'] = 0.
    data['beta'] = 0.
    # topology = Topology(np.unique(data[static_columns]))
    topology = Topology(data[static_columns].drop_duplicates())

    # Next create Trajectory (the result)
    # n_atoms = len(np.unique(data['atom_id'].values))
    n_atoms = data['atom_id'].nunique()
    # result = Trajectory(structured_to_unstructured(data[dynamical_columns]).reshape(n_structures, n_atoms, 3),
    #                     topology=topology)
    result = Trajectory(data[dynamical_columns].values.reshape(n_structures, n_atoms, 3), topology=topology)

    # Return
    return result

# @jit(nopython=False)
# def _atom_reader(buffer):
#     result = ''
#     for line in buffer.readlines():
#         if line[:4] == 'ATOM':
#             result += line
#     return result
