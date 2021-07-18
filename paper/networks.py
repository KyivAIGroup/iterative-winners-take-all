from kwta import iWTA, iWTA_history, kWTA
from permanence import *


__all__ = [
    "NetworkWillshaw",
    "NetworkKWTA",
    "NetworkPermanenceFixedSparsity",
    "NetworkPermanenceVogels",
    "NetworkPermanenceVaryingSparsity"
]


class Network:
    def __init__(self, weights: dict, weights_learn, perm_cls: type):
        self.weights = weights
        self.weights_learn = weights_learn
        for name in self.weights_learn:
            self.weights[name] = perm_cls(self.weights[name])


class NetworkWillshaw(Network):
    name = "Classical Willshaw iWTA"

    def __init__(self, weights: dict, weights_learn=()):
        super().__init__(weights, weights_learn, perm_cls=ParameterBinary)

    def train_epoch(self, x, n_choose=10, **kwargs):
        h, y = iWTA(x, **self.weights)
        if 'w_xy' in self.weights_learn:
            self.weights['w_xy'].update(x_pre=x, x_post=y, n_choose=n_choose)
        if 'w_xh' in self.weights_learn:
            self.weights['w_xh'].update(x_pre=x, x_post=h, n_choose=n_choose)
        if 'w_hh' in self.weights_learn:
            self.weights['w_hh'].update(x_pre=h, x_post=h, n_choose=n_choose)
        if 'w_hy' in self.weights_learn:
            self.weights['w_hy'].update(x_pre=h, x_post=y, n_choose=n_choose)
        if 'w_yy' in self.weights_learn:
            self.weights['w_yy'].update(x_pre=y, x_post=y, n_choose=n_choose)
        if 'w_yh' in self.weights_learn:
            self.weights['w_yh'].update(x_pre=y, x_post=h, n_choose=n_choose)
        return h, y


class NetworkKWTA(Network):
    name = "kWTA permanence fixed sparsity"
    K_FIXED = 10

    def __init__(self, weights: dict, weights_learn=()):
        weights_learn = set(weights_learn).intersection(['w_xy', 'w_xh', 'w_yh'])
        super().__init__(weights, weights_learn, perm_cls=PermanenceFixedSparsity)

    def train_epoch(self, x, n_choose=10, lr=0.001):
        h = kWTA(self.weights['w_xh'] @ x, k=self.K_FIXED)
        y = kWTA(self.weights['w_xy'] @ x - self.weights['w_hy'] @ h, k=self.K_FIXED)
        if 'w_xy' in self.weights_learn:
            self.weights['w_xy'].update(x_pre=x, x_post=y, n_choose=n_choose, lr=lr)
        if 'w_xh' in self.weights_learn:
            self.weights['w_xh'].update(x_pre=x, x_post=h, n_choose=n_choose, lr=lr)
        if 'w_hy' in self.weights_learn:
            self.weights['w_hy'].update(x_pre=h, x_post=y, n_choose=n_choose)
        return h, y


class NetworkPermanenceFixedSparsity(Network):
    name = "Permanence fixed sparsity"

    def __init__(self, weights: dict, weights_learn=()):
        super().__init__(weights, weights_learn, perm_cls=PermanenceFixedSparsity)

    def train_epoch(self, x, n_choose=10, lr=0.001):
        h, y = iWTA(x, **self.weights)
        if 'w_xy' in self.weights_learn:
            self.weights['w_xy'].update(x_pre=x, x_post=y, n_choose=n_choose, lr=lr)
        if 'w_xh' in self.weights_learn:
            self.weights['w_xh'].update(x_pre=x, x_post=h, n_choose=n_choose, lr=lr)
        if 'w_hh' in self.weights_learn:
            self.weights['w_hh'].update(x_pre=h, x_post=h, n_choose=n_choose, lr=lr)
        if 'w_hy' in self.weights_learn:
            self.weights['w_hy'].update(x_pre=h, x_post=y, n_choose=n_choose, lr=lr)
        if 'w_yy' in self.weights_learn:
            self.weights['w_yy'].update(x_pre=y, x_post=y, n_choose=n_choose, lr=lr)
        if 'w_yh' in self.weights_learn:
            self.weights['w_yh'].update(x_pre=y, x_post=h, n_choose=n_choose, lr=lr)
        return h, y


class NetworkPermanenceVaryingSparsity(NetworkPermanenceFixedSparsity):
    name = "Permanence varying sparsity"

    def __init__(self, weights: dict, weights_learn=(),
                 output_sparsity_desired=(0.025, 0.1)):
        self.weights = weights
        self.weights_learn = weights_learn
        excitatory = ('w_xh', 'w_xy', 'w_yh', 'w_yy')
        for name in self.weights_learn:
            self.weights[name] = PermanenceVaryingSparsity(
                self.weights[name],
                excitatory=name in excitatory,
                output_sparsity_desired=output_sparsity_desired
            )


class NetworkPermanenceVogels(Network):
    name = "Permanence Vogels"

    def __init__(self, weights: dict, weights_learn=()):
        self.weights = weights
        self.weights_learn = weights_learn
        for name in ("w_xy", "w_xh", "w_yy", "w_yh"):
            # Excitatory weights must be updated by some other rule
            if name in weights_learn:
                self.weights[name] = PermanenceFixedSparsity(weights[name])
        for name in ("w_hy", "w_hh"):
            # Inhibitory weights are updated by Vogels
            if name in weights_learn:
                self.weights[name] = PermanenceVogels(weights[name])

    def train_epoch(self, x, n_choose=10, lr=0.001):
        z_h, z_y = iWTA_history(x, **self.weights)
        h, y = z_h[0], z_y[0]
        for i in range(1, len(z_h)):
            h |= z_h[i]
            y |= z_y[i]

        if 'w_xy' in self.weights_learn:
            self.weights['w_xy'].update(x_pre=x, x_post=y, n_choose=n_choose, lr=lr)
        if 'w_xh' in self.weights_learn:
            self.weights['w_xh'].update(x_pre=x, x_post=h, n_choose=n_choose, lr=lr)
        if 'w_hh' in self.weights_learn:
            self.weights['w_hh'].update(x_pre=z_h, x_post=z_h, n_choose=n_choose, lr=lr)
        if 'w_hy' in self.weights_learn:
            self.weights['w_hy'].update(x_pre=z_h, x_post=z_y, n_choose=n_choose, lr=lr)
        if 'w_yy' in self.weights_learn:
            self.weights['w_yy'].update(x_pre=y, x_post=y, n_choose=n_choose, lr=lr)
        if 'w_yh' in self.weights_learn:
            self.weights['w_yh'].update(x_pre=y, x_post=h, n_choose=n_choose, lr=lr)

        return h, y
