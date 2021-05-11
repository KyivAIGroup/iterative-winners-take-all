import warnings
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def kWTA(x, k):
    # x is a (N,) vec or (N, trials) tensor
    if k == 0:
        return np.zeros_like(x)
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    winners = np.argsort(x, axis=0)[-k:]
    sdr = np.zeros_like(x)
    sdr[winners, range(x.shape[1])] = 1
    return sdr.squeeze()


def kWTAi_doesnt_work(x, w_lat):
    y = np.zeros_like(x, dtype=np.int32)
    for threshold in range(np.max(x), 0, -1):
        z = x - w_lat @ y >= threshold
        y = np.logical_or(y, z)
    return y


def overlap(x1, x2):
    return (x1 & x2).sum()


def cosine_similarity(x1, x2):
    return x1.dot(x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def kWTAi(y0, h0, w_hy, w_yy=None, w_hh=None, w_yh=None):
    # y0 is a (Ny,) vec or (Ny, trials) tensor
    # h0 is a (Nh,) vec or (Nh, trials) tensor
    h = np.zeros_like(h0, dtype=np.int32)
    y = np.zeros_like(y0, dtype=np.int32)
    t_start = max(np.max(h0), np.max(y0))
    for threshold in range(t_start, 0, -1):
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

    # TODO the same hack should be for 'h'
    if y.ndim == 1:
        y = np.expand_dims(y, axis=1)
    empty_cols = ~(y.any(axis=0))
    if empty_cols.any():
        warnings.warn("kWTAi resulted in a zero vector. "
                      "Activating one neuron manually.")
        y_kwta = kWTA(y0 - w_hy @ h0, k=1)
        if y_kwta.ndim == 1:
            y_kwta = np.expand_dims(y_kwta, axis=1)
        y[:, empty_cols] = y_kwta[:, empty_cols]
    y = y.squeeze()

    return h, y


def update_weights(w, x_pre, x_post, n_choose=1):
    inds_pre = np.random.choice(np.nonzero(x_pre)[0], n_choose)
    inds_post = np.random.choice(np.nonzero(x_post)[0], n_choose)
    w[inds_post, inds_pre] = 1
