"""
geometry.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""


import numpy as np


# Compute angle between three points
def angle(a, b, c, method='atan2'):
    r"""
    Compute angle between three points :math:`\angle ABC`.

    Parameters
    ----------
    a, b, c : ArrayLike
        Cartesian coordinates
    method : str
        Method to compute angle, see :func:`~vangle` for options.

    Returns
    -------
    float or numpy.ndarray
        Angle between points
    """

    u = vector(a, b)
    v = vector(b, c)

    return vangle(u, v, method=method)


# Convert Cartesian to polar coordinates
def cartesian_to_polar(a):
    """
    Convert Cartesian to polar coordinates.

    Parameters
    ----------
    a : ArrayLike
        Cartesian coordinates.

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
    r"""
    Compute cosine of the angle between three points :math:`\angle ABC`.

    Parameters
    ----------
    a, b, c : ArrayLike
        Vectors

    Returns
    -------
    float or numpy.ndarray
        Cosine angle of points
    """

    u = vector(a, b)
    v = vector(b, c)

    return cos_vangle(u, v)


# Compute angle between vectors
def cos_vangle(a, b):
    """
    Compute cosine angle between vectors.

    Parameters
    ----------
    a, b : ArrayLike
        Vectors

    Returns
    -------
    float or numpy.ndarray
    """

    # Coerce
    a, b, needs_ravel = _coerce_to_2d(a, b)

    # Return
    return _array_result(dot(a, b) / (norm(a) * norm(b)), needs_ravel)


# Cross product
def cross(a, b):
    """
    Compute cross product between vectors `a` and `b`. Points to :func:`numpy.cross`

    Parameters
    ----------
    a, b : ArrayLike
        Vectors

    Returns
    -------
    float or numpy.ndarray
        Cross product of u and v
    """

    return np.cross(a, b)


# Dihedral angle between 4 points
def dihedral(a, b, c, d):
    """
    Compute dihedral angle between four points.

    Parameters
    ----------
    a, b, c, d : ArrayLike
        Cartesian coordinates.

    Returns
    -------
    float or numpy.ndarray
        Dihedral angle
    """

    _check_dimensions(a, b, c, d, n_dim=3)

    u = vector(a, b)
    v = vector(b, c)
    w = vector(c, d)

    return vdihedral(u, v, w)


# Compute the distance between two vectors
def distance(a, b=None):
    """
    Compute the Euclidean distance between two vectors

    Parameters
    ----------
    a, b : ArrayLike
        Vectors

    Returns
    -------
    float
        Distance
    """

    # Coerce
    a, needs_ravel = _coerce_to_2d(a)

    # If y is not supplied, set to zeros; then coerce
    if b is None:
        b = np.zeros(a.shape)
    b, _ = _coerce_to_2d(b)

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
    float or numpy.ndarray
        Vector dot product
    """

    a, b, needs_ravel = _coerce_to_2d(a, b)

    return _array_result(np.sum(np.multiply(a, b), axis=axis), needs_ravel)


# Normed vector
def norm(a):
    r"""
    Compute vector norm.

    .. math :: |a| = \sqrt{a_x^2 + a_y^2 + a_z^2}

    Parameters
    ----------
    a : ArrayLike
        Vector

    Returns
    -------
    float or numpy.ndarray
        Normed vector
    """

    # a, needs_ravel = _coerce_to_2d(a)
    #
    # return _array_result(np.linalg.norm(a, axis=1), needs_ravel)
    return distance(a)


# Compute the normal between three points
def normal(a, b, c):
    """
    Compute normal vector between three points.

    Parameters
    ----------
    a, b, c : ArrayLike
        Cartesian coordinates.

    Returns
    -------
    numpy.ndarray
        Vector normal
    """

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
    float or numpy.ndarray
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
    r"""
    Compute unit vector.

    .. math :: \^{a} = \frac{a}{|a|}

    Parameters
    ----------
    a : ArrayLike
        Vector

    Returns
    -------
    numpy.ndarray
        Unit vector
    """

    a, needs_ravel = _coerce_to_2d(a)
    return _array_result(a / norm(a).reshape(-1, 1), needs_ravel)


# Compute angle between 2 vectors
def vangle(a, b, method='atan2'):
    r"""
    Compute the angle between two vectors.

    If method = 'atan2',

    ..math :: \theta = atan2(norm(cross(a, b)), dot(a, b))

    If method = 'acos',

    ..math :: \theta = acos \frac{a \dot b}{|a| |b|}

    Parameters
    ----------
    a, b : ArrayLike
        Vectors
    method : str
        Using 'atan2' or 'acos' method of computing angle.

    Returns
    -------
    float or numpy.ndarray
        Angle between vectors
    """

    method = str(method).lower()

    if method == 'atan2':
        cross_ = cross(a, b)
        if _has_dimensions(a, b, n_dim=2):
            if a.ndim == 2 and b.ndim == 2:
                cross_ = cross_.reshape(-1, 1)
            else:
                cross_ = [cross_]
        result = np.arctan2(norm(cross_), dot(a, b))
    elif method == 'acos':
        result = np.arccos(cos_vangle(a, b))
    else:
        raise AttributeError('method %s unknown' % method)

    return result


# Compute dihedral between 3 vectors
def vdihedral(a, b, c):
    """
    Compute the dihedral angle between three vectors.

    Parameters
    ----------
    a, b, c : ArrayLike
        Vectors

    Returns
    -------
    float or numpy.ndarray
        Dihedral angle
    """

    _check_dimensions(a, b, c, n_dim=3)

    u = vnormal(a, b)
    v = vnormal(b, c)

    return vangle(u, v)


# Compute vector between 2 sets of points
def vector(a, b, normalize=False):
    """
    Compute vector between two sets of points.

    Parameters
    ----------
    a, b : ArrayLike
        Cartesian coordinates.
    normalize : bool
        Should the unit vector be computed? (Default: False)

    Returns
    -------
    numpy.ndarray
        Vector between `a` and `b`.
    """

    # Coerce input
    a, b, needs_ravel = _coerce_to_2d(a, b)

    v = np.subtract(b, a)
    if normalize:
        v /= norm(v).reshape(-1, 1)

    return _array_result(v, needs_ravel)


# Compute normal
def vnormal(a, b):
    """
    Compute the normal between two vectors.

    Parameters
    ----------
    a, b : ArrayLike
        Vectors

    Returns
    -------
    numpy.ndarray
        Vector normal
    """

    return cross(a, b)


# Helper function to ravel array result
def _array_result(a, needs_ravel=False):
    if needs_ravel:
        if a.ndim > 1:
            a = a.ravel()
        else:
            a = a[0]
    return a


def _check_dimensions(*args, n_dim=3):
    if not _has_dimensions(*args, n_dim=n_dim):
        raise AttributeError('must be %sD' % n_dim)


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


# Helper function to check dimensionality
def _has_dimensions(*args, n_dim=3):
    result = []
    for arg in args:
        result.append((arg.ndim > 1 and arg.shape[1] == n_dim) or (arg.ndim == 1 and arg.shape[0] == n_dim))
    return all(result)
