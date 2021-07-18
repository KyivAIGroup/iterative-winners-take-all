from kwta import iWTA, iWTA_history, kWTA
from permanence import *


__all__ = [
    "NetworkWillshaw",
    "NetworkKWTA",
    "NetworkPermanenceFixedSparsity",
    "NetworkPermanenceVogels",
    "NetworkPermanenceVaryingSparsity"
]


class NetworkWillshaw:
    name = "Classical Willshaw iWTA"

    def __init__(self, weights: dict):
        self.weights = {}
        for name, w in weights.items():
            self.weights[name] = ParameterBinary(w)

    def train_epoch(self, x, n_choose=10, **kwargs):
        h, y = iWTA(x, **self.weights)
        self.weights['w_xy'].update(x_pre=x, x_post=y, n_choose=n_choose)
        self.weights['w_xh'].update(x_pre=x, x_post=h, n_choose=n_choose)
        self.weights['w_hh'].update(x_pre=h, x_post=h, n_choose=n_choose)
        self.weights['w_hy'].update(x_pre=h, x_post=y, n_choose=n_choose)
        if self.weights['w_yy'] is not None:
            self.weights['w_yy'].update(x_pre=y, x_post=y, n_choose=n_choose)
        self.weights['w_yh'].update(x_pre=y, x_post=h, n_choose=n_choose)
        return h, y


class NetworkKWTA:
    name = "kWTA permanence fixed sparsity"
    K_FIXED = 10

    def __init__(self, weights: dict):
        self.weights = {}
        self.weights['w_xh'] = PermanenceFixedSparsity(weights['w_xh'])
        self.weights['w_xy'] = PermanenceFixedSparsity(weights['w_xy'])
        self.weights['w_hy'] = weights['w_hy']

    def train_epoch(self, x, n_choose=10, lr=0.001):
        h = kWTA(self.weights['w_xh'] @ x, k=self.K_FIXED)
        y = kWTA(self.weights['w_xy'] @ x - self.weights['w_hy'] @ h, k=self.K_FIXED)
        self.weights['w_xy'].update(x_pre=x, x_post=y, n_choose=n_choose, lr=lr)
        self.weights['w_xh'].update(x_pre=x, x_post=h, n_choose=n_choose, lr=lr)
        return h, y


class NetworkPermanenceFixedSparsity:
    name = "Permanence fixed sparsity"

    def __init__(self, weights: dict):
        self.weights = {}
        for name, w in weights.items():
            self.weights[name] = PermanenceFixedSparsity(w)

    def train_epoch(self, x, n_choose=10, lr=0.001):
        h, y = iWTA(x, **self.weights)
        self.weights['w_xy'].update(x_pre=x, x_post=y, n_choose=n_choose, lr=lr)
        self.weights['w_xh'].update(x_pre=x, x_post=h, n_choose=n_choose, lr=lr)
        self.weights['w_hh'].update(x_pre=h, x_post=h, n_choose=n_choose, lr=lr)
        self.weights['w_hy'].update(x_pre=h, x_post=y, n_choose=n_choose, lr=lr)
        if self.weights['w_yy'] is not None:
            self.weights['w_yy'].update(x_pre=y, x_post=y, n_choose=n_choose, lr=lr)
        self.weights['w_yh'].update(x_pre=y, x_post=h, n_choose=n_choose, lr=lr)
        return h, y


class NetworkPermanenceVaryingSparsity(NetworkPermanenceFixedSparsity):
    name = "Permanence varying sparsity"

    def __init__(self, weights: dict, output_sparsity_desired=(0.025, 0.1)):
        self.weights = {}
        self.weights['w_xh'] = PermanenceVaryingSparsity(weights['w_xh'],
                                                         excitatory=True,
                                                         output_sparsity_desired=output_sparsity_desired)
        self.weights['w_xy'] = PermanenceVaryingSparsity(weights['w_xy'],
                                                         excitatory=True,
                                                         output_sparsity_desired=output_sparsity_desired)
        self.weights['w_hh'] = PermanenceVaryingSparsity(weights['w_hh'],
                                                         excitatory=False,
                                                         output_sparsity_desired=output_sparsity_desired)
        self.weights['w_hy'] = PermanenceVaryingSparsity(weights['w_hh'],
                                                         excitatory=False,
                                                         output_sparsity_desired=output_sparsity_desired)
        self.weights['w_yy'] = PermanenceVaryingSparsity(weights['w_yy'],
                                                         excitatory=True,
                                                         output_sparsity_desired=output_sparsity_desired)
        self.weights['w_yh'] = PermanenceVaryingSparsity(weights['w_yh'],
                                                         excitatory=True,
                                                         output_sparsity_desired=output_sparsity_desired)


class NetworkPermanenceVogels:
    name = "Permanence Vogels"

    def __init__(self, weights: dict):
        self.weights = {}
        for name in ("w_xy", "w_xh", "w_yy", "w_yh"):
            self.weights[name] = PermanenceFixedSparsity(weights[name])
        for name in ("w_hy", "w_hh"):
            self.weights[name] = PermanenceVogels(weights[name])

    def train_epoch(self, x, n_choose=10, lr=0.001):
        z_h, z_y = iWTA_history(x, **self.weights)
        h, y = z_h[0], z_y[0]
        for i in range(1, len(z_h)):
            h |= z_h[i]
            y |= z_y[i]

        self.weights['w_xy'].update(x_pre=x, x_post=y, n_choose=n_choose, lr=lr)
        self.weights['w_xh'].update(x_pre=x, x_post=h, n_choose=n_choose, lr=lr)
        self.weights['w_hh'].update(x_pre=z_h, x_post=z_h, n_choose=n_choose, lr=lr)
        self.weights['w_hy'].update(x_pre=z_h, x_post=z_y, n_choose=n_choose, lr=lr)
        if self.weights['w_yy'] is not None:
            self.weights['w_yy'].update(x_pre=y, x_post=y, n_choose=n_choose, lr=lr)
        self.weights['w_yh'].update(x_pre=y, x_post=h, n_choose=n_choose, lr=lr)

        return h, y
