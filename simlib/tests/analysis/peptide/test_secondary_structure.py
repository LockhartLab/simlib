"""
test_secondary_structure.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""


from simlib.analysis.peptide import compute_secondary_structure
from simlib.io import read_pdb

import numpy as np
from unittest import TestCase


class TestSecondaryStructure(TestCase):
    # Test that we can compute the secondary structure
    def test_secondary_structure(self):
        # Read in trajectory
        trajectory = read_pdb('simlib/tests/samples/trajectory_short.pdb')

        # Compute secondary structure
        secondary_structure = compute_secondary_structure(trajectory, executable='C:/Users/Chris/Downloads/stride/stride.exe')
        print(secondary_structure.residues_with_code('C').block_averages())

        # print(ss.mean(axis=None))
        # print(ss.mean(axis=0))
        # print(ss.mean(axis=1))

