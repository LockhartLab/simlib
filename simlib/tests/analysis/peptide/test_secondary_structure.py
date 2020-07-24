"""
test_secondary_structure.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""


from simlib.analysis.protein import secondary_structure
from simlib.io import read_pdb

import numpy as np
import os.path
from unittest import TestCase

# Directory to samples
samples = os.path.join('simlib', 'tests', 'samples')


class TestSecondaryStructure(TestCase):
    # Test that we can compute the secondary structure
    def test_secondary_structure(self):
        # Read in trajectory
        trajectory = read_pdb(os.path.join(samples, 'trajectory.pdb'))

        # Compute secondary structure
        ss = secondary_structure(trajectory, executable='C:/Users/Chris/Downloads/stride/stride.exe')
        print(ss.residues_with_code('C').mean(axis=0))

        # print(ss.mean(axis=None))
        # print(ss.mean(axis=0))
        # print(ss.mean(axis=1))

