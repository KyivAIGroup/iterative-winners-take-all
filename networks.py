"""
Implementation of iterative winners-take-all (iWTA) and k-winners-take-all
(kWTA) networks.
"""

from kwta import iWTA, iWTA_history, kWTA
from permanence import *


__all__ = [
    "NetworkSimpleHebb",
    "NetworkKWTA",
    "NetworkPermanenceFixedSparsity",
    "NetworkPermanenceVogels",
    "NetworkPermanenceVaryingSparsity"
]


class Network:
    """
    A base class for all the networks below.
    """
    def __init__(self, weights: dict, weights_learn, perm_cls: type):
        self.weights = weights
        self.weights_learn = weights_learn
        for name in self.weights_learn:
            self.weights[name] = perm_cls(self.weights[name])


class NetworkSimpleHebb(Network):
    """
    An iWTA network that takes input 'x', outputs inhibitory 'h' and
    excitatory 'y' signals and is updated by the classical Willshaw's
    associative memory rule:

    w_ij = 1, if y_i = 1 and x_j = 1.

    Required connections: w_xh, w_xy, w_hy.
    Optional connections: w_hh, w_yy, w_yh.

    Parameters
    ----------
    weights : dict
        A dictionary with connection names and matrix values.
    weights_learn : tuple or list of str
        A list of connection names to learn. The rest are set fixed.
    """

    name = "SimpleHebb"

    def __init__(self, weights: dict, weights_learn=()):
        super().__init__(weights, weights_learn, perm_cls=SimpleHebb)

    def train_epoch(self, x, n_choose=10, **kwargs):
        """
        Train one full iteration (an epoch) on all samples at once.

        Parameters
        ----------
        x : (Nx, S) np.ndarray
            Input vectors tensor. The first axis is neurons, and the second is
            the sample (trial) ID.
        n_choose : int, optional
            Non-zero values to choose to update from x-h and x-y outer
            products.
            Default: 10

        Returns
        -------
        h : (Nh, S) np.ndarray
            Inhibitory populations output.
        y : (Ny, S) np.ndarray
            Excitatory populations output.
        """
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
    """
    A kWTA network that takes input 'x', outputs inhibitory 'h' and
    excitatory 'y' and is updated by the permanence-fixed learning rule.

    h = kWTA(w_xh @ x, 10)
    y = kWTA(w_xy @ x - w_hy @ h, 10)

    Required connections: w_xh, w_xy, w_hy.

    Parameters
    ----------
    weights : dict
        A dictionary with connection names and matrix values.
    weights_learn : tuple or list of str
        A list of connection names to learn. The rest are set fixed.
    """

    name = "kWTA + permanence fixed weight sparsity"
    K_FIXED = 10

    def __init__(self, weights: dict, weights_learn=()):
        weights_learn = set(weights_learn).intersection(['w_xy', 'w_xh', 'w_yh'])
        super().__init__(weights, weights_learn, perm_cls=PermanenceFixedSparsity)

    def train_epoch(self, x, n_choose=10, lr=0.001):
        """
        Train one full iteration (an epoch) on all samples at once.

        Parameters
        ----------
        x : (Nx, S) np.ndarray
            Input vectors tensor. The first axis is neurons, and the second is
            the sample (trial) ID.
        n_choose : int, optional
            Non-zero values to choose to update from x-h and x-y outer
            products.
            Default: 10
        lr : float, optional
            The learning rate.
            Default: 0.001

        Returns
        -------
        h : (Nh, S) np.ndarray
            Inhibitory populations output.
        y : (Ny, S) np.ndarray
            Excitatory populations output.
        """
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
    """
    An iWTA network that takes input 'x', outputs inhibitory 'h' and
    excitatory 'y' and is updated by the permanence-fixed learning rule.

    Required connections: w_xh, w_xy, w_hy.
    Optional connections: w_hh, w_yy, w_yh.

    Parameters
    ----------
    weights : dict
        A dictionary with connection names and matrix values.
    weights_learn : tuple or list of str
        A list of connection names to learn. The rest are set fixed.
    """

    name = "Permanence fixed weight sparsity"

    def __init__(self, weights: dict, weights_learn=()):
        super().__init__(weights, weights_learn, perm_cls=PermanenceFixedSparsity)

    def train_epoch(self, x, n_choose=10, lr=0.001):
        """
        Train one full iteration (an epoch) on all samples at once.

        Parameters
        ----------
        x : (Nx, S) np.ndarray
            Input vectors tensor. The first axis is neurons, and the second is
            the sample (trial) ID.
        n_choose : int, optional
            Non-zero values to choose to update from x-h and x-y outer
            products.
            Default: 10
        lr : float, optional
            The learning rate.
            Default: 0.001

        Returns
        -------
        h : (Nh, S) np.ndarray
            Inhibitory populations output.
        y : (Ny, S) np.ndarray
            Excitatory populations output.
        """
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
    """
    An iWTA network that takes input 'x', outputs inhibitory 'h' and
    excitatory 'y' and is updated by the permanence-varying learning rule.

    Required connections: w_xh, w_xy, w_hy.
    Optional connections: w_hh, w_yy, w_yh.

    Parameters
    ----------
    weights : dict
        A dictionary with connection names and matrix values.
    weights_learn : tuple or list of str
        A list of connection names to learn. The rest are set fixed.
    """

    name = "Permanence varying weight sparsity"

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
    """
    An iWTA network that takes input 'x', outputs inhibitory 'h' and
    excitatory 'y'. Inhibitory w_hy and w_hh connections are updated by Vogels'
    learning rule, excitatory - by the permanence-fixed model.

    Required connections: w_xh, w_xy, w_hy.
    Optional connections: w_hh, w_yy, w_yh.

    Parameters
    ----------
    weights : dict
        A dictionary with connection names and matrix values.
    weights_learn : tuple or list of str
        A list of connection names to learn. The rest are set fixed.
    """

    name = "Permanence update by Vogels"

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
        """
        Train one full iteration (an epoch) on all samples at once.

        Parameters
        ----------
        x : (Nx, S) np.ndarray
            Input vectors tensor. The first axis is neurons, and the second is
            the sample (trial) ID.
        n_choose : int, optional
            Non-zero values to choose to update from x-h and x-y outer
            products.
            Default: 10
        lr : float, optional
            The learning rate.
            Default: 0.001

        Returns
        -------
        h : (Nh, S) np.ndarray
            Inhibitory populations output.
        y : (Ny, S) np.ndarray
            Excitatory populations output.
        """
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
