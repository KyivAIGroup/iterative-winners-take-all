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
    # y0 is a (Ny, trials) tensor
    # h0 is a (Nh, trials) tensor
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


def update_weights(w, x_pre, x_post, n_choose=1):
    # x_pre and x_post are (N, trials) tensors
    assert x_pre.shape[1] == x_post.shape[1]
    for x, y in zip(x_pre.T, x_post.T):
        x = x.nonzero()[0]
        y = y.nonzero()[0]
        if n_choose is None:
            # full outer product
            w[np.expand_dims(y, axis=1), x] = 1
        else:
            # a subset of the outer product
            x = np.random.choice(x, n_choose)
            y = np.random.choice(y, n_choose)
            w[y, x] = 1


def normalize_presynaptic(mat):
    presum = mat.sum(axis=1)[:, np.newaxis]
    presum += 1e-10
    mat /= presum


class PermanenceFixedSparsity(np.ndarray):

    def __new__(cls, data):
        assert np.unique(data).tolist() == [0, 1], "A binary matrix is expected"
        mat = np.array(data, dtype=np.int32).view(cls)
        permanence = data * np.random.random(data.shape)
        normalize_presynaptic(permanence)
        mat.permanence = permanence
        return mat

    def __array_finalize__(self, obj):
        self.permanence = getattr(obj, 'permanence', None)

    @property
    def n_active(self):
        return self.sum()

    def update(self, x_pre, x_post, n_choose=1, lr=0.001):
        for x, y in zip(x_pre.T, x_post.T):
            x = x.nonzero()[0]
            y = y.nonzero()[0]
            if n_choose is None:
                # full outer product
                self.permanence[np.expand_dims(y, axis=1), x] += lr
            else:
                # a subset of the outer product
                x = np.random.choice(x, size=n_choose)
                y = np.random.choice(y, size=n_choose)
                self.permanence[y, x] += lr
        self.normalize()

    def normalize(self):
        # self.permanence.clip(min=0, out=self.permanence)
        normalize_presynaptic(self.permanence)
        data = kWTA(self.permanence.reshape(-1), k=self.n_active)
        self[:] = data.reshape(self.shape)


class PermanenceWithDropout(PermanenceFixedSparsity):

    def __new__(cls, data, excitatory: bool, sparsity_desired=(0.025, 0.1)):
        mat = super().__new__(cls, data)
        mat.excitatory = excitatory
        mat.sparsity_desired = sparsity_desired
        mat.dropout = np.random.random()
        return mat

    def update_dropout(self, output_sparsity: float, gamma=0.1):
        dropout_inc = gamma * 0.95 + (1 - gamma) * self.dropout
        dropout_dec = gamma * 0.05 + (1 - gamma) * self.dropout
        dropout = self.dropout
        s_min, s_max = self.sparsity_desired
        if output_sparsity > s_max:
            dropout = dropout_inc if self.excitatory else dropout_dec
        elif output_sparsity < s_min:
            dropout = dropout_dec if self.excitatory else dropout_inc
        return dropout

    def update(self, x_pre, x_post, n_choose=1, lr=0.001):
        output_sparsity = np.count_nonzero(x_post) / x_post.size
        self.dropout = self.update_dropout(output_sparsity)
        super().update(x_pre, x_post, n_choose=n_choose, lr=lr)

    def normalize(self):
        normalize_presynaptic(self.permanence)
        perm = self.permanence.reshape(-1)
        perm = perm[perm > 0]
        pmax = perm.max()
        perm = perm[perm != pmax]  # the max element should survive
        n_active = len(perm)
        if n_active > 0:
            # find k-th smallest value
            perm = np.sort(perm)
            n_drop = max(1, int(self.dropout * n_active))
            threshold = perm[-n_drop]
        else:
            # pick any value in (0, pmax) range exclusively
            threshold = 0.5 * pmax
        self.permanence[self.permanence <= threshold] = 0
        self[:] = self.permanence > 0
