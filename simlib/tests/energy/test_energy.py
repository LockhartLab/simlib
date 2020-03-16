"""
test_energy.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""


from simlib.energy import *

from hypothesis import given
from hypothesis import strategies as st
import numpy as np
from unittest import TestCase


class TestEnergy(TestCase):
    @given(st.integers(min_value=1, max_value=1000),
           st.integers(min_value=1, max_value=3),
           st.floats(min_value=0., max_value=100.),
           st.floats(min_value=0., max_value=100.))
    def test_bond(self, n_atoms, n_dim, r0, k):
        # Create points
        a = np.random.rand(n_atoms, n_dim)
        b = np.random.rand(n_atoms, n_dim)
        c = np.random.rand(n_atoms, n_dim)
        d = np.random.rand(n_atoms, n_dim)

        # Get vectors
        u = b - a
        v = c - b
        w = d - c

        # Sanity
        r = np.sqrt(np.sum(np.square(b - a), axis=1))
        x = (b - a) / r[:, None]
        y = np.array([(b - a)[:, i] / r for i in range(a.shape[1])]).T
        np.testing.assert_array_equal(x.ravel(), np.array(y).ravel())

        # Bond
        r = np.sqrt(np.sum(np.square(b - a), axis=1))
        u0 = 0.5 * k * (r - r0)**2
        du0 = k * (r - r0)[:, None] * (b - a) / r[:, None]
        bond = Bond(a, b, r0, k)
        u1 = bond.energy
        np.testing.assert_almost_equal(u0, u1)
        # np.testing.assert_array_almost_equal(du0.ravel(), np.array(du1).ravel())

        # Angle
        r = np.arccos(np.vdot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        u0 = 0.5 * k * (r - r0) ** 2
        # du0 = k * (r - r0)[:, None] * (b - a) / r[:, None]
        angle = Angle(a, b, c, r0, k)
        u1 = angle.energy
        np.testing.assert_almost_equal(u0, u1)
        # np.testing.assert_array_almost_equal(du0.ravel(), np.array(du1).ravel())
