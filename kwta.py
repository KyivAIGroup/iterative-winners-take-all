"""
Iterative winners-take-all and k-winners-take-all activation functions.
"""

import numpy as np


def kWTA(x, k):
    """
    Default k-winners-take-all activation function.

    Parameters
    ----------
    x : np.ndarray
        A presynaptic sum (N,) vector or (N, S) array of vector samples.
    k : int
        The number of active neurons in the output.

    Returns
    -------
    sdr : np.ndarray
        A binary vector or array with the same shape as `x` and exactly `k`
        neurons active.
    """
    if k == 0:
        return np.zeros_like(x)
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    winners = np.argsort(x, axis=0)[-k:]  # (k, S) shape
    sdr = np.zeros_like(x)
    sdr[winners, range(x.shape[1])] = 1
    return sdr.squeeze()


def iWTA(x, w_xh, w_xy, w_hy, w_yy=None, w_hh=None, w_yh=None):
    """
    Iterative winners-take-all activation function.

    Parameters
    ----------
    x : (Nx, S) np.ndarray
        The input samples.
    w_xh, w_xy, w_hy, w_yy, w_hh, w_yh : np.ndarray or None
        Binary weights.

    Returns
    -------
    h : (Nh, S) np.ndarray
        Inhibitory populations output.
    y : (Ny, S) np.ndarray
        Excitatory populations output.
    """
    h0 = w_xh @ x
    y0 = w_xy @ x
    h = np.zeros_like(h0, dtype=np.int32)
    y = np.zeros_like(y0, dtype=np.int32)
    t_start = max(h0.max(), y0.max())  # <= 1.0
    for threshold in np.linspace(t_start, 0, num=20, endpoint=False):
        z_h = h0
        if w_hh is not None:
            z_h = z_h - w_hh @ h
        if w_yh is not None:
            z_h = z_h + w_yh @ y
        z_h = z_h >= threshold

        z_y = y0 - w_hy @ h
        if w_yy is not None:
            z_y += w_yy @ y
        z_y = z_y >= threshold

        h |= z_h
        y |= z_y

    return h, y


def iWTA_history(x, w_xh, w_xy, w_hy, w_yy=None, w_hh=None, w_yh=None):
    """
    Iterative winners-take-all activation function history. Used in Vogels'
    update rule.

    Parameters
    ----------
    x : (Nx, S) np.ndarray
        The input samples.
    w_xh, w_xy, w_hy, w_yy, w_hh, w_yh : np.ndarray or None
        Binary weights.

    Returns
    -------
    z_h, z_y : tuple
        Tuples of size `S`, each containing intermediate activations of
        inhibitory and excitatory populations that were balancing each other.
    """
    h0 = w_xh @ x
    y0 = w_xy @ x
    h = np.zeros_like(h0, dtype=np.int32)
    y = np.zeros_like(y0, dtype=np.int32)
    t_start = max(h0.max(), y0.max())  # <= 1.0
    history = []
    for threshold in np.linspace(t_start, 0, num=20, endpoint=False):
        z_h = h0
        if w_hh is not None:
            z_h = z_h - w_hh @ h
        if w_yh is not None:
            z_h = z_h + w_yh @ y
        z_h = z_h >= threshold

        z_y = y0 - w_hy @ h
        if w_yy is not None:
            z_y += w_yy @ y
        z_y = z_y >= threshold

        h |= z_h
        y |= z_y
        history.append((z_h, z_y))

    z_h, z_y = zip(*history)

    return z_h, z_y
