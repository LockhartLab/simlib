"""
geometry.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""


import numpy as np


# Compute angle between three points
def angle(a, b, c):
    u = vector(a, b)
    v = vector(b, c)
    return vangle(u, v)


# Convert Cartesian to polar coordinates
def cartesian_to_polar(a):
    """
    Convert Cartesian to polar coordinates

    Parameters
    ----------
    a : ArrayLike
        Cartesian coordinates

    Returns
    -------
    numpy.ndarray
        Polar coordinates in same shape as `a`
    """

    # Coerce input
    a, needs_ravel = _coerce_to_2d(a)

    n_dim = a.shape[1]
    if n_dim in (2, 3):
        r = norm(a)
        result = [r, np.arctan2(a[:, 1], a[:, 0])]
        if n_dim == 3:
            result.append(np.arccos(a[:, 2] / r))

    else:
        raise AttributeError('cannot compute for %s dimensions' % n_dim)

    return _array_result(np.array(result).T, needs_ravel)


# Compute angle between three points
def cos_angle(a, b, c):
    u = vector(a, b)
    v = vector(b, c)
    return cos_vangle(u, v)


def cos_vangle(u, v):
    u, v, needs_ravel = _coerce_to_2d(u, v)
    return _array_result(dot(u, v) / (norm(u) * norm(v)), needs_ravel)


# Dihedral angle between 4 points
def dihedral(a, b, c, d):
    if (a.ndim > 1 and a.shape[1] != 3) or (a.ndim == 1 and a.shape[0] != 3):
        raise AttributeError('must be 3D')
    u = vector(a, b)
    v = vector(b, c)
    w = vector(c, d)
    return vdihedral(u, v, w)


# Compute the distance between two vectors
def distance(a, b=None, method='euclidean'):
    """
    Compute the distance between two vectors

    Parameters
    ----------
    a : ArrayLike
    b : ArrayLike

    method : str
        (Default: 'euclidean')

    Returns
    -------
    float
        Distance
    """

    # Coerce
    a, b, needs_ravel = _coerce_to_2d(a, b)

    # If y is not supplied, set to zeros
    if b is None:
        b = np.zeros(a.shape)

    # Return distance
    return _array_result(np.sqrt(np.sum(np.square(vector(a, b)), axis=1)), needs_ravel)


# Dot product
def dot(a, b, axis=1):
    """
    Compute vector dot product.

    Parameters
    ----------
    a, b : ArrayLike
        Vectors
    axis : int
        Axis to apply sum over.

    Returns
    -------
    numpy.ndarray
        Vector dot product
    """

    return np.sum(np.multiply(a, b), axis=axis)


# Normed vector
def norm(u):
    u, needs_ravel = _coerce_to_2d(u)
    return _array_result(np.linalg.norm(u, axis=1), needs_ravel)


# Compute the normal between four points
def normal(a, b, c):
    u = vector(a, b)
    v = vector(b, c)
    return vnormal(u, v)


# Polar to cartesian coordinates
def polar_to_cartesian(a):
    """
    Convert polar to Cartesian coordinates.

    Parameters
    ----------
    a : ArrayLike
        Polar coordinates

    Returns
    -------
    numpy.ndarray
        Cartesian coordinates, same shape as `a`
    """

    # Coerce
    a, needs_ravel = _coerce_to_2d(a)

    n_dim = a.shape[1]
    if n_dim in (2, 3):
        result = np.array([np.cos(a[:, 1]), np.sin(a[:, 1])])
        if n_dim == 3:
            result *= np.sin(a[:, 2])
            result = np.vstack([result, np.cos(a[:, 2])])
        result *= a[:, 0]

    else:
        raise AttributeError('cannot compute for %s dimensions' % n_dim)

    return _array_result(np.array(result).T, needs_ravel)


# Create unit vector
def uvector(a):
    a, needs_ravel = _coerce_to_2d(a)
    return _array_result(a / norm(a).reshape(-1, 1), needs_ravel)


# Compute angle between 2 vectors
def vangle(u, v):
    return np.arccos(cos_vangle(u, v))


# Compute dihedral between 3 vectors
def vdihedral(u, v, w):
    if (u.ndim > 1 and u.shape[1] != 3) or (u.ndim == 1 and u.shape[0] != 3):
        raise AttributeError('must be 3D')
    q = vnormal(u, v)
    r = vnormal(v, w)
    return vangle(q, r)


# Compute vector between 2 sets of points
def vector(a, b, normalize=False):
    # Coerce input
    a, b, needs_ravel = _coerce_to_2d(a, b)

    v = np.subtract(b, a)
    if normalize:
        v /= norm(v).reshape(-1, 1)

    return _array_result(v, needs_ravel)


# Compute normal
def vnormal(u, v):
    return np.cross(u, v)


def _array_result(a, needs_ravel=False):
    if needs_ravel:
        if a.ndim > 1:
            a = a.ravel()
        else:
            a = a[0]
    return a


def _coerce_to_2d(a, *args):
    # Coerce a to 2d
    a = np.array(a)
    needs_ravel = False
    if a.ndim == 1:
        needs_ravel = True
        a = a.reshape(1, -1)
    result = [a]

    # Do the same for kwargs
    for arg in args:
        b, _ = _coerce_to_2d(arg)
        if a.shape != b.shape:
            raise AttributeError('vectors must be same shapes')
        result.append(b)

    # Add in needs_ravel flag
    result.append(needs_ravel)

    # Return
    return result
