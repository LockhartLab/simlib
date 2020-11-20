"""
time_series.py

author: C. Lockhart <chris@lockhartlab.org>
language: Python3
"""

import numpy as np


# noinspection PyShadowingNames
def acorr(a):
    r"""
    Compute the lagged autocorrelation of an observation :math:`a`.

    .. math :: \rho(\tau) = \sum_

    Note: this is mostly derived from :func:`matplotlib.axes.Axes.acorr`.

    Parameters
    ----------
    a : numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Autocorrelation function
    """

    # We only want to compute this for 1D arrays
    if a.ndim != 1:
        raise AttributeError('must be 1D')

    # numpy.correlate with mode='full' computes the un-normalized correlation from -len(a) to len(a)
    # To normalize this, we must divide by np.sqrt(np.dot(a, a) * np.dot(a, a)) = np.dot(a, a)
    # Then, we are only interested in the data from lag = 0:len(a), so we subset the resulting array
    rho = (np.correlate(a, a, mode='full') / np.dot(a, a))[(len(a) - 1):]

    # As a sanity check, index 0 must be 1.
    if not np.allclose(rho[0], 1.):
        raise ValueError('autocorrelation function not computed correctly')

    # Return
    return rho


# noinspection PyShadowingNames
def _acorr(a):
    n = len(a)
    u = np.mean(a)
    s = np.sum(np.square(a - u))
    rho = []
    for i in np.arange(len(a)):
        rho.append(np.sum((a[:n - i] - u) * (a[i:] - u)) / s)
    return rho


def _acorr_test(a, decimal=7):
    rho0 = acorr(a)
    rho1 = _acorr(a)
    np.testing.assert_almost_equal(rho0, rho1, decimal=decimal)


# noinspection PyShadowingNames
def inefficiency(a):
    """
    Compute the statistical inefficiency :math:`g` from the equilibration time \tau_{eq}.

    .. math :: g = 1 + 2\tau_{eq}.

    Parameters
    ----------
    a : numpy.ndarray

    Returns
    -------
    float
        Statistical inefficiency.
    """

    return 1. + 2. * teq(a)


# Estimate the standard error from the correlation time
# noinspection PyShadowingNames
def sem_tcorr(a, tol=1e-3):
    """
    Estimate of standard error of the mean derived from the correlation time.

    The main assumption is that sqrt(N) ~ sqrt(len(a) / tcorr(a))

    Should only be used for continuous time series data, e.g., from molecular dynamics. Discontinuous trajectories as
    produced by replica exchange or Monte Carlo are not applicable.

    Parameters
    ----------
    a : numpy.ndarray
    tol : float

    Returns
    -------

    """

    return np.std(a) * np.sqrt(tcorr(a, tol=tol) / len(a))


# TODO how is this different than teq?
# noinspection PyShadowingNames
def tcorr(a, tol=1e-3):
    """
    Compute correlation time from the autocorrelation function.

    Parameters
    ----------
    a : numpy.ndarray
    tol : float

    Returns
    -------
    int
        Index of `a` where first converged, i.e., always less than `tol`.
    """

    # Autocorrelation function
    # noinspection PyShadowingNames
    rho = acorr(a)

    # Accumulate the maximum for all time points
    # Strictly, this is performed on the reversed and absolute autocorrelation function rho (i.e., looking backwards)
    rho_rev_max = np.maximum.accumulate(np.abs(rho[::-1]))

    # Find rho_rev_max < tol
    rho_rev_equil = rho_rev_max < tol

    # If there are no instances where we are equilibrated,
    if np.sum(rho_rev_equil) == 0:
        raise ValueError(f'autocorrelation not converged with tol={tol}')

    # Return N - first observed convergence
    return len(rho) - np.argmin(rho_rev_equil)


# noinspection PyShadowingNames
def _tcorr(a, tol=1e-3):
    """
    Compute correlation time from the autocorrelation function.

    Parameters
    ----------
    a
    tol

    Returns
    -------

    """

    rho = acorr(a)

    for t in np.arange(len(rho)):
        if np.max(np.abs(rho[t:])) < tol:
            return t

    raise AttributeError(f'no correlation time found with tol={tol}')


def _tcorr_test(a, tol=1e-3):
    tau0 = tcorr(a, tol=tol)
    tau1 = _tcorr(a, tol=tol)
    np.testing.assert_equal(tau0, tau1)


# noinspection PyShadowingNames
def teq(a):
    r"""
    Compute equilibration time :math:`\tau_{eq}` based on the autocorrelation function :math:`\rho(t)`.

    .. math :: \tau_{eq} = \sum_{t=1}^{T} (1 - \frac{t}{T}) \rho_{t}

    Parameters
    ----------
    a : numpy.ndarray

    Returns
    -------
    float
        Equilibration time.
    """

    rho = acorr(a)[1:]
    t_max = len(a)
    t = np.arange(1, t_max)

    return np.sum((1. - t / t_max) * np.abs(rho))



if __name__ == '__main__':
    n = 100000
    a = np.random.normal(loc=0, scale=10, size=n)
    print(teq(a))
    # _acorr_test(a, decimal=3)
    # _tcorr_test(a)

    # print(np.std(a))
    # print(np.sqrt(np.cov(a)))
    # print(np.sqrt(xx - x*x))
    # print(correlation_time(a))
    # print(sem_tcorr(a))

    # import time
    # start_time = time.time()
    # rho = acorr(a)
    # end_time = time.time()
    # print(end_time - start_time)
    # start_time = time.time()
    # tau = tcorr(a)
    # end_time = time.time()
    # print(end_time - start_time)
