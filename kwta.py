import numpy as np
import warnings


def kWTA(x, k):
    winners = np.argsort(x)[-k:]
    sdr = np.zeros_like(x)
    sdr[winners] = 1
    return sdr


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
            z_h -= w_hh @ h
        if w_yh is not None:
            z_h += w_yh @ y
        z_h = z_h >= threshold

        z_y = y0 - w_hy @ h
        if w_yy is not None:
            z_y += w_yy @ y
        z_y = z_y >= threshold

        h |= z_h
        y |= z_y

    if not y.any():
        # TODO the same hack should be for 'h'
        warnings.warn("kWTAi resulted in a zero vector. "
                      "Activating one neuron manually.")
        y = kWTA(y0 - w_hy @ h0, k=1)

    return h, y


def update_weights(w, x_pre, x_post, n_choose=1):
    inds_pre = np.random.choice(np.nonzero(x_pre)[0], n_choose)
    inds_post = np.random.choice(np.nonzero(x_post)[0], n_choose)
    w[inds_post, inds_pre] = 1
