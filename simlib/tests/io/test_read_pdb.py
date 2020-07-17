"""
test_read_pdb.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from simlib.io import read_pdb
# from simlib.framework import Structure, Trajectory
from simlib.framework import Trajectory

import numpy as np
import os.path
from unittest import TestCase

# Path to samples
samples = os.path.join('simlib', 'tests', 'samples')


# Test that we know how to read PDB files
class TestReadPDB(TestCase):
    # Test that we know how to read a single PDB
    def test_read_structure(self):
        # Try to read a single PDB
        trajectory = read_pdb(os.path.join(samples, 'structure.pdb'))

        # Must be of type Structure
        self.assertIsInstance(trajectory, Trajectory)

        # Number of frames
        self.assertEqual(len(trajectory), 1)
        self.assertEqual(trajectory.n_structures, 1)

        # Number of atoms
        self.assertEqual(trajectory.n_atoms, 15354)
        self.assertEqual(len(trajectory.topology), 15354)

        # Number of dimensions
        self.assertEqual(trajectory.n_dim, 3)

    # Test that we know how to read a PDB trajectory
    def test_read_trajectory(self):
        # Read trajectory
        trajectory = read_pdb(os.path.join(samples, 'trajectory.pdb'))

        # Must be of type Trajectory
        self.assertIsInstance(trajectory, Trajectory)

        # Number of frames
        self.assertEqual(len(trajectory), 5)
        self.assertEqual(trajectory.n_structures, 5)

        # Number of atoms
        self.assertEqual(trajectory.n_atoms, 41806)
        self.assertEqual(len(trajectory.topology), 41806)

        # Number of dimensions
        self.assertEqual(trajectory.n_dim, 3)
