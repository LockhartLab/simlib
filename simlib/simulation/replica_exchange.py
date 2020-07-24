import numpy as np


def replica_temperatures(t0=300, dt=1.01, n_replicas=40, mode='geometric'):
    """
    Geometric = the ratio from one temperature to the next is kept constant


    Parameters
    ----------
    t0
    dt
    n_replicas
    mode

    Returns
    -------

    """
    if mode != 'geometric':
        raise AttributeError('only supports geometric')

    return t0 * np.power(dt, np.arange(n_replicas))
