"""
test_geometry.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""


from simlib.geometry.geometry import *

from hypothesis import given
from hypothesis import strategies as st
import numpy as np
from unittest import TestCase


class TestGeometry(TestCase):
    @given(st.integers(min_value=0, max_value=1000),
           st.integers(min_value=2, max_value=3))
    def test_points(self, n_points, n_dim):
        # Shape
        if n_points > 0:
            shape = [n_points, n_dim]
        else:
            shape = [n_dim]

        # Get points
        a = np.random.rand(*shape)
        b = np.random.rand(*shape)
        c = np.random.rand(*shape)
        d = np.random.rand(*shape)

        # Compute vectors
        u = b - a
        v = c - b
        w = d - c

        # Test vectors
        np.testing.assert_equal(u, vector(a, b))
        np.testing.assert_equal(v, vector(b, c))
        np.testing.assert_equal(w, vector(c, d))

        # Compute normalized vectors
        u_norm = _norm(u, n_points)
        v_norm = _norm(v, n_points)
        w_norm = _norm(w, n_points)

        # Test normalized vectors
        np.testing.assert_equal(u_norm.ravel(), norm(u))
        np.testing.assert_equal(v_norm.ravel(), norm(v))
        np.testing.assert_equal(w_norm.ravel(), norm(w))
        np.testing.assert_equal(u / u_norm, vector(a, b, normalize=True))
        np.testing.assert_equal(v / v_norm, vector(b, c, normalize=True))
        np.testing.assert_equal(w / w_norm, vector(c, d, normalize=True))

        # Get unit vectors
        u_unit = u / u_norm
        v_unit = v / v_norm
        w_unit = w / w_norm

        # Test unit vectors
        np.testing.assert_equal(u_unit, uvector(u))
        np.testing.assert_equal(v_unit, uvector(v))
        np.testing.assert_equal(w_unit, uvector(w))

        # Get cos angles
        cos_q = _dot(u_unit, v_unit, n_points)
        cos_r = _dot(v_unit, w_unit, n_points)

        # Test cos angles
        np.testing.assert_almost_equal(cos_q, cos_angle(a, b, c))
        np.testing.assert_almost_equal(cos_r, cos_angle(b, c, d))

        # Compute angles
        q = np.arccos(cos_q)
        r = np.arccos(cos_r)

        # Test angles
        np.testing.assert_almost_equal(q, angle(a, b, c, method='acos'))
        np.testing.assert_almost_equal(r, angle(b, c, d, method='acos'))
        np.testing.assert_almost_equal(q, vangle(u, v, method='acos'))
        np.testing.assert_almost_equal(r, vangle(v, w, method='acos'))
        np.testing.assert_almost_equal(q, angle(a, b, c, method='atan2'))
        np.testing.assert_almost_equal(r, angle(b, c, d, method='atan2'))
        np.testing.assert_almost_equal(q, vangle(u, v, method='atan2'))
        np.testing.assert_almost_equal(r, vangle(v, w, method='atan2'))
        with np.testing.assert_raises(AttributeError):
            vangle(u, v, method='junk')

        # Compute normal
        np.testing.assert_equal(np.cross(u, v), normal(a, b, c))
        np.testing.assert_equal(np.cross(v, w), normal(b, c, d))
        np.testing.assert_equal(np.cross(u, v), vnormal(u, v))
        np.testing.assert_equal(np.cross(v, w), vnormal(v, w))

        # Compute dihedral
        if n_dim == 3:
            phi = vangle(np.cross(u, v), np.cross(v, w))
            np.testing.assert_equal(phi, dihedral(a, b, c, d))
            np.testing.assert_equal(phi, vdihedral(u, v, w))
        else:
            with np.testing.assert_raises(AttributeError):
                dihedral(a, b, c, d)
            with np.testing.assert_raises(AttributeError):
                vdihedral(u, v, w)

        # Compute distance
        np.testing.assert_equal(_distance(a, b, n_points), distance(a, b))
        np.testing.assert_equal(_distance(b, c, n_points), distance(b, c))
        np.testing.assert_equal(_distance(c, d, n_points), distance(c, d))

        # Compute xyz to polar and vice versa (don't need to compute because we can use identities)
        np.testing.assert_array_almost_equal(a, polar_to_cartesian(cartesian_to_polar(a)))
        np.testing.assert_array_almost_equal(a, cartesian_to_polar(polar_to_cartesian(a)))
        with np.testing.assert_raises(AttributeError):
            cartesian_to_polar(np.random.rand(3, 1))
        with np.testing.assert_raises(AttributeError):
            polar_to_cartesian(np.random.rand(3, 1))

        # Misc
        with np.testing.assert_raises(AttributeError):
            vector(np.random.rand(3, 2), np.random.rand(3, 1))


def _distance(a, b, n_points):
    if n_points == 0:
        result = np.sqrt(np.sum(np.square(b - a)))
    else:
        result = np.sqrt(np.sum(np.square(b - a), axis=1))
    return result


def _dot(u, v, n_points):
    if n_points == 0:
        result = np.vdot(u, v)
    else:
        result = np.sum(np.multiply(u, v), axis=1)
    return result


def _norm(u, n_points):
    if n_points == 0:
        result = np.linalg.norm(u)
    elif n_points == 1:
        result = np.linalg.norm(u, axis=1)
    else:
        result = np.linalg.norm(u, axis=1).reshape(-1, 1)
    return result
