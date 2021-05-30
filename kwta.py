import warnings

import numpy as np


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


def kWTA_different_k(x_tensor, ks):
    # x_tensor is a (N, trials) tensor
    # ks is a list of (trials,) the num. of the top 'k' neurons
    assert x_tensor.shape[1] == len(ks)
    argsort = np.argsort(x_tensor, axis=0)[::-1]
    sdr = np.zeros_like(x_tensor)
    for trial_id in range(x_tensor.shape[1]):
        k = ks[trial_id]
        winners = argsort[:k, trial_id]
        sdr[winners, trial_id] = 1
        assert (sdr[:, trial_id] == kWTA(x_tensor[:, trial_id], k=k)).all()
    return sdr


def iWTA(y0, h0, w_hy, w_yy=None, w_hh=None, w_yh=None):
    # y0 is a (Ny,) vec or (Ny, trials) tensor
    # h0 is a (Nh,) vec or (Nh, trials) tensor
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

    # TODO the same hack should be for 'h'
    if y.ndim == 1:
        y = np.expand_dims(y, axis=1)
    empty_trials = ~(y.any(axis=0))
    if empty_trials.any():
        # This is particularly wrong because y != y_kwta even when k=1
        warnings.warn("iWTA resulted in a zero vector. "
                      "Activating one neuron manually.")
        h_kwta = kWTA(h0, k=1)
        y_kwta = kWTA(y0 - w_hy @ h_kwta, k=1)
        if y_kwta.ndim == 1:
            h_kwta = np.expand_dims(h_kwta, axis=1)
            y_kwta = np.expand_dims(y_kwta, axis=1)
        h[:, empty_trials] = h_kwta[:, empty_trials]
        y[:, empty_trials] = y_kwta[:, empty_trials]
    h = h.squeeze()
    y = y.squeeze()

    return h, y


def update_weights(w, x_pre, x_post, n_choose=1):
    if x_pre.ndim == 2:
        assert x_pre.shape[1] == x_post.shape[1]
        for x, y in zip(x_pre.T, x_post.T):
            update_weights(w, x_pre=x, x_post=y)
        return
    x_pre_idx = x_pre.nonzero()[0]
    if len(x_pre_idx) == 0:
        warnings.warn("'x_pre' is a zero vector")
        return
    x_post_idx = x_post.nonzero()[0]
    if len(x_post_idx) == 0:
        warnings.warn("'x_post' is a zero vector")
        return
    idx_pre = np.random.choice(x_pre_idx, n_choose)
    idx_post = np.random.choice(x_post_idx, n_choose)
    w[idx_post, idx_pre] = 1
