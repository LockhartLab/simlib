"""
test_protein.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from simlib.analysis.protein import hydrophobic_moment_from_sequence

from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import unittest

# Contents
__all__ = [
    'TestProtein'
]


# Test protein.py
class TestProtein(unittest.TestCase):
    def test_dummy(self):
        self.assertTrue(True)
