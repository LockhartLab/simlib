"""
test_geometry.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""


from simlib.geometry import *

# from hypothesis import given
# from hypothesis import strategies as st
import numpy as np
from pytest import fixture


@fixture
def load_vertices():
    a = np.random.rand(100, 3)
    b = np.random.rand(100, 3)
    c = np.random.rand(100, 3)
    return a, b, c


# Test angle
def test_angle(load_vertices):
    # Load vertices
    a, b, c = load_vertices

    # Wrapper to easily perform testing
    # noinspection PyShadowingNames
    def _tester(a, b, c=None):
        u, v = np.subtract(a, b), np.subtract(c, b)
        _angle = np.arccos(dot(u, v) / (norm(u) * norm(v)))
        np.testing.assert_array_almost_equal(_angle, angle(a, b, c))
        np.testing.assert_array_almost_equal(_angle, angle(a, b, c, method='acos'))

    # Two singulars as vectors will throw an error
    with np.testing.assert_raises(AttributeError):
        angle(a[0, 0], b[0, 0])

    # Three singulars as vertices will throw an error
    with np.testing.assert_raises(AttributeError):
        angle(a[0, 0], b[0, 0], c[0, 0])

    # 2D
    _tester(a[0, :2], b[0, :2], c[0, :2])
    _tester(a[:, :2], b[:, :2], c[:, :2])

    # 3D
    _tester(a[0, :], b[0, :], c[0, :])
    _tester(a[:, :], b[:, :], c[:, :])


# Test