"""
test_write_pdb.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from simlib.io import read_pdb

import numpy as np
import os.path
from unittest import TestCase

# Path to samples
samples = os.path.join('simlib', 'tests', 'samples')


# Test that we know how to read PDB files
class TestWritePDB(TestCase):
    # Test that we know how to read a single PDB
    def test_write_structure(self):
        # Try to read a single PDB
        trajectory1 = read_pdb(os.path.join(samples, 'structure.pdb'))

        # Write as PDB
        trajectory1.to_pdb('temp.pdb')

        # Read written PDB back in
        trajectory2 = read_pdb('temp.pdb')

        # Make sure that the two trajectories match
        xyz1 = trajectory1.xyz
        xyz2 = trajectory2.xyz
        np.testing.assert_array_equal(xyz1, xyz2)

    # Test that we know how to read a PDB trajectory
    def test_write_trajectory(self):
        # Read trajectory
        trajectory1 = read_pdb(os.path.join(samples, 'trajectory.pdb'))

        # Write as PDB
        trajectory1.to_pdb('temp.pdb')

        # Read written PDB back in
        trajectory2 = read_pdb('temp.pdb')

        # Make sure that the two trajectories match
        xyz1 = trajectory1.xyz
        xyz2 = trajectory2.xyz
        np.testing.assert_array_equal(xyz1, xyz2)
