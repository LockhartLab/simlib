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


# TODO: in general, tests should be broken into 2 categories. First, will it run? Second, is it right?
class TestGeometry(TestCase):
    def test_execution(self):
        pass

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
        u = a - b
        v = b - c
        w = c - d

        # Test vectors
        np.testing.assert_equal(u, vector(a, b))
        np.testing.assert_equal(v, vector(b, c))
        np.testing.assert_equal(w, vector(c, d))

        # Compute normalized vectors
        u_norm = _norm(u, n_points)
        v_norm = _norm(v, n_points)
        w_norm = _norm(w, n_points)

        # Test normalized vectors
        np.testing.assert_almost_equal(u_norm.ravel(), norm(u).reshape(-1))
        np.testing.assert_almost_equal(v_norm.ravel(), norm(v).reshape(-1))
        np.testing.assert_almost_equal(w_norm.ravel(), norm(w).reshape(-1))
        np.testing.assert_almost_equal(u / u_norm, vector(a, b, normalize=True))
        np.testing.assert_almost_equal(v / v_norm, vector(b, c, normalize=True))
        np.testing.assert_almost_equal(w / w_norm, vector(c, d, normalize=True))

        # Get unit vectors
        u_unit = u / u_norm
        v_unit = v / v_norm
        w_unit = w / w_norm

        # Test unit vectors
        np.testing.assert_equal(u_unit, unit_vector(u))
        np.testing.assert_equal(v_unit, unit_vector(v))
        np.testing.assert_equal(w_unit, unit_vector(w))

        # Get cos angles
        cos_q = _dot((a - b) / _norm(a - b, n_points), (c - b) / _norm(c - b, n_points), n_points)
        cos_r = _dot((a - b) / _norm(a - b, n_points), (d - b) / _norm(d - b, n_points), n_points)
        cos_s = _dot((a - c) / _norm(a - c, n_points), (d - c) / _norm(d - c, n_points), n_points)
        cos_t = _dot((b - c) / _norm(b - c, n_points), (d - c) / _norm(d - c, n_points), n_points)

        # Test cos angles
        np.testing.assert_almost_equal(cos_q, cos_angle(a, b, c))
        np.testing.assert_almost_equal(cos_r, cos_angle(a, b, d))
        np.testing.assert_almost_equal(cos_s, cos_angle(a, c, d))
        np.testing.assert_almost_equal(cos_t, cos_angle(b, c, d))

        # Compute angles
        q = np.arccos(cos_q)
        r = np.arccos(cos_r)
        s = np.arccos(cos_s)
        t = np.arccos(cos_t)

        # Test unsigned angles using acos
        np.testing.assert_almost_equal(q, angle(a, b, c, method='acos'))  #, signed=False))
        np.testing.assert_almost_equal(r, angle(a, b, d, method='acos'))  #, signed=False))
        np.testing.assert_almost_equal(s, angle(a, c, d, method='acos'))  #, signed=False))
        np.testing.assert_almost_equal(t, angle(b, c, d, method='acos'))  #, signed=False))

        # Test unsigned angles using atan2
        np.testing.assert_almost_equal(q, angle(a, b, c, method='atan2'))  #, signed=False))
        np.testing.assert_almost_equal(r, angle(a, b, d, method='atan2'))  #, signed=False))
        np.testing.assert_almost_equal(s, angle(a, c, d, method='atan2'))  #, signed=False))
        np.testing.assert_almost_equal(t, angle(b, c, d, method='atan2'))  #, signed=False))

        # TODO add test for signed angle

        # Test vertex angle vs vector angle
        np.testing.assert_almost_equal(angle(vector(a, b), vector(c, b)), angle(a, b, c))
        np.testing.assert_almost_equal(angle(vector(a, b), vector(d, b)), angle(a, b, d))
        np.testing.assert_almost_equal(angle(vector(a, c), vector(d, c)), angle(a, c, d))
        np.testing.assert_almost_equal(angle(vector(b, c), vector(d, c)), angle(b, c, d))

        # Test that method attribute works
        with np.testing.assert_raises(AttributeError):
            angle(a, b, c, method='junk')

        # Test normal
        np.testing.assert_equal(np.cross(a - b, c - b), normal(a, b, c))
        np.testing.assert_equal(np.cross(a - b, d - b), normal(a, b, d))
        np.testing.assert_equal(np.cross(a - c, d - c), normal(a, c, d))
        np.testing.assert_equal(np.cross(b - c, d - c), normal(b, c, d))

        # Test vertex normal vs vector normal
        np.testing.assert_equal(normal(vector(a, b), vector(c, b)), normal(a, b, c))
        np.testing.assert_equal(normal(vector(a, b), vector(d, b)), normal(a, b, d))
        np.testing.assert_equal(normal(vector(a, c), vector(d, c)), normal(a, c, d))
        np.testing.assert_equal(normal(vector(b, c), vector(d, c)), normal(b, c, d))

        # Compute dihedral
        # if n_dim == 3:
        #     phi = vangle(np.cross(u, v), np.cross(v, w))
        #     np.testing.assert_equal(phi, dihedral(a, b, c, d))
        #     np.testing.assert_equal(phi, vdihedral(u, v, w))
        # else:
        #     with np.testing.assert_raises(AttributeError):
        #         dihedral(a, b, c, d)
        #     with np.testing.assert_raises(AttributeError):
        #         vdihedral(u, v, w)

        # Compute distance
        np.testing.assert_equal(_distance(a, b, n_points), distance(a, b))
        np.testing.assert_equal(_distance(b, c, n_points), distance(b, c))
        np.testing.assert_equal(_distance(c, d, n_points), distance(c, d))

        # Test xyz to polar and vice versa (don't need to compute because we can use identities)
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
