import numpy as np
import warnings


def kWTA(x, k):
    # x is a (N,) vec or (N, trials) tensor
    if k == 0:
        return np.zeros_like(x)
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    winners = np.argsort(x, axis=0)[-k:]  # (k, trials)
    sdr = np.zeros_like(x)
    sdr[winners, range(x.shape[1])] = 1
    return sdr.squeeze()


def at_least_one_neuron_active(y0, h0, y, h, w_hy):
    empty_trials = ~(y.any(axis=0))
    if empty_trials.any():
        warnings.warn("iWTA resulted in a zero vector. "
                      "Activating one neuron manually.")
        h_kwta = kWTA(h0, k=1)
        y_kwta = kWTA(y0 - w_hy @ h_kwta, k=1)
        h[:, empty_trials] = h_kwta[:, empty_trials]
        y[:, empty_trials] = y_kwta[:, empty_trials]
    return h, y


def iWTA(x, w_xh, w_xy, w_hy, w_yy=None, w_hh=None, w_yh=None):
    h0 = w_xh @ x
    y0 = w_xy @ x
    h = np.zeros_like(h0, dtype=np.int32)
    y = np.zeros_like(y0, dtype=np.int32)
    t_start = max(h0.max(), y0.max())
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

    h, y = at_least_one_neuron_active(y0, h0, y, h, w_hy)

    return h, y


def iWTA_history(x, w_xh, w_xy, w_hy, w_yy=None, w_hh=None, w_yh=None):
    h0 = w_xh @ x
    y0 = w_xy @ x
    h = np.zeros_like(h0, dtype=np.int32)
    y = np.zeros_like(y0, dtype=np.int32)
    t_start = max(h0.max(), y0.max())
    history = []
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
        history.append((z_h, z_y))

    z_h, z_y = zip(*history)
    z_h = list(z_h)
    z_y = list(z_y)
    z_h[-1], z_y[-1] = at_least_one_neuron_active(y0, h0, z_y[-1], z_h[-1], w_hy)

    return z_h, z_y


def update_weights(w, x_pre, x_post, n_choose=1):
    # x_pre and x_post are (N, trials) tensors
    assert x_pre.shape[1] == x_post.shape[1]
    for x, y in zip(x_pre.T, x_post.T):
        x = x.nonzero()[0]
        y = y.nonzero()[0]
        if len(x) == 0 or len(y) == 0:
            continue
        if n_choose is None or n_choose >= len(x) * len(y):
            # full outer product
            w[np.expand_dims(y, axis=1), x] = 1
        else:
            # a subset of the outer product
            x = np.random.choice(x, n_choose)
            y = np.random.choice(y, n_choose)
            w[y, x] = 1
