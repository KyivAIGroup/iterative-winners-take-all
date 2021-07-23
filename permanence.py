"""
Learning rules implementation:
  1. Classical Willshaw associative memory (SimpleHebb).
  2. Permanence with fixed sparsity.
  3. Permanence updates by Vogels (not shown in the paper).
  4. Permanence with varying sparsity.
"""

import math
import numpy as np

from kwta import kWTA


__all__ = [
    "SimpleHebb",
    "PermanenceFixedSparsity",
    "PermanenceVogels",
    "PermanenceVaryingSparsity"
]


def normalize_presynaptic(mat):
    """
    Normalize the presynpatic sum to have a unit norm for each input neuron
    individually.

    Parameters
    ----------
    mat : np.ndarray
        Real-valued connection weights (permanences).
    """
    presum = mat.sum(axis=1)[:, np.newaxis]
    presum += 1e-10  # avoid division by zero
    mat /= presum


class SimpleHebb(np.ndarray):
    """
    A weight matrix with classical Willshaw associative memory learning rule:

    w_ij = 1, if y_i = 1 and x_j = 1.
    """

    def __new__(cls, data, **kwargs):
        """
        Convert a numpy array to the SimpleHebb instance.

        Parameters
        ----------
        data : (N_output, N_input) np.ndarray
            Initial binary connections weights.
        """
        if data is None:
            return None
        assert np.unique(data).tolist() == [0, 1], "A binary matrix is expected"
        mat = np.array(data, dtype=np.int32).view(cls)
        return mat

    def update(self, x_pre, x_post, n_choose=10, **kwargs):
        """
        Update the connections weights from the pre- and post-synaptic
        activations.

        Parameters
        ----------
        x_pre, x_post : np.ndarray
            Pre- and post-synaptic activations. It's a 2D array, the first
            axis is neurons, and the second is the sample (trial) ID. The
            dimensionality of the fist axis can differ for the pre- and post-
            synaptic vectors.
        n_choose : int, optional
            Non-zero values to choose to update from the pre- and post- outer
            products.
            Default: 10
        """
        assert x_pre.shape[1] == x_post.shape[1], "Batch size mismatch"
        for x, y in zip(x_pre.T, x_post.T):
            x = x.nonzero()[0]
            y = y.nonzero()[0]
            if len(x) == 0 or len(y) == 0:
                continue
            if n_choose is None or n_choose >= len(x) * len(y):
                # full outer product
                self[np.expand_dims(y, axis=1), x] = 1
            else:
                # a subset of the outer product
                x = np.random.choice(x, n_choose)
                y = np.random.choice(y, n_choose)
                self[y, x] = 1

    def __matmul__(self, matrix):
        # treat as a numpy array when multiplied by a matrix
        return self.view(np.ndarray) @ matrix


class PermanenceFixedSparsity(SimpleHebb):
    """
    A weight matrix with permanence with fixed sparsity learning rule.
    The binary weights sparsity is kept fixed and equal to the initial sparsity
    throughout learning.
    """

    def __new__(cls, data, **kwargs):
        """
        Convert a numpy array to the PermanenceFixedSparsity instance.

        Parameters
        ----------
        data : (N_output, N_input) np.ndarray
            Initial binary connections weights.
        """
        if data is None:
            return None
        assert np.unique(data).tolist() == [0, 1], "A binary matrix is expected"
        mat = data.view(cls)
        permanence = np.random.random(data.shape)
        normalize_presynaptic(permanence)
        mat.permanence = permanence
        return mat

    def __array_finalize__(self, obj):
        self.permanence = getattr(obj, 'permanence', None)

    def update(self, x_pre, x_post, n_choose=10, lr=0.001):
        """
        Update the connections weights from the pre- and post-synaptic
        activations.

        Parameters
        ----------
        x_pre, x_post : np.ndarray
            Pre- and post-synaptic activations. It's a 2D array, the first
            axis is neurons, and the second is the sample (trial) ID. The
            dimensionality of the fist axis can differ for the pre- and post-
            synaptic vectors.
        n_choose : int, optional
            Non-zero values to choose to update from the pre- and post- outer
            products.
            Default: 10
        lr : float, optional
            The learning rate.
            Default: 0.001
        """
        assert x_pre.shape[1] == x_post.shape[1], "Batch size mismatch"
        for x, y in zip(x_pre.T, x_post.T):
            x = x.nonzero()[0]
            y = y.nonzero()[0]
            if len(x) == 0 or len(y) == 0:
                continue
            if n_choose is None or n_choose >= len(x) * len(y):
                # full outer product
                self.permanence[np.expand_dims(y, axis=1), x] += lr
            else:
                # a subset of the outer product
                x = np.random.choice(x, size=n_choose)
                y = np.random.choice(y, size=n_choose)
                self.permanence[y, x] += lr
        self.normalize()

    def normalize(self):
        """
        Normalize the permanence and binary weights matrices.
        """
        normalize_presynaptic(self.permanence)
        # Each output neuron will have 'k' synapses to input neurons.
        # Keep the weight sparsity fixed.
        k = math.ceil(self.sum() / self.shape[0])

        # Leave the 'k' largest entries of 'P' in 'w' for each output neuron
        winners = np.argsort(self.permanence, axis=1)[:, -k:]  # (N_out, k)
        self.fill(0)
        self[np.arange(self.shape[0])[:, np.newaxis], winners] = 1


class PermanenceVogels(PermanenceFixedSparsity):
    """
    An inhibitory weight matrix with permanence and update rule by Vogels.
    """

    def update(self, x_pre, x_post, n_choose=10, lr=0.001,
               neighbors_coincident=1):
        """
        Update the connections weights from the pre- and post-synaptic
        activations.

        Parameters
        ----------
        x_pre, x_post : np.ndarray
            Pre- and post-synaptic activations. It's a 2D array, the first
            axis is neurons, and the second is the sample (trial) ID. The
            dimensionality of the fist axis can differ for the pre- and post-
            synaptic vectors.
        n_choose : int, optional
            Non-zero values to choose to update from the pre- and post- outer
            products.
            Default: 10
        lr : float, optional
            The learning rate.
            Default: 0.001
        neighbors_coincident : int, optional
            The number of coincident steps (subiterations of an iWTA model)
            to potentiate in a window fashion.
            Default: 1
        """
        assert len(x_pre) == len(x_post)
        window_size = (neighbors_coincident + 1)
        n_steps = len(x_pre)
        # multiply by ~10 to be on a similar learning time scale with
        # other learning rules
        lr_potentiation = 10 * lr / (n_steps * window_size)
        lr_depression = -10 * lr / (n_steps * (n_steps - window_size))
        for i, (x, y) in enumerate(zip(x_pre, x_post)):
            for j in range(max(0, i - neighbors_coincident), i + 1):
                # Potentiation
                x_recent = x_pre[j]
                alpha = lr_potentiation * (1 - (i - j) / window_size)
                super().update(x_pre=x_recent, x_post=y, lr=alpha)
            for j in range(0, i - neighbors_coincident - 1):
                # Depression
                x_past = x_pre[j]
                super().update(x_pre=x_past, x_post=y, lr=lr_depression)
        self.normalize_vogels()

    def normalize(self):
        # Called each time super().update() is executed.
        # Don't normalize intermediate z_h and z_y updates.
        pass

    def normalize_vogels(self):
        """
        Normalize the permanence and binary weights matrices.
        """
        self.permanence.clip(min=0, out=self.permanence)
        super().normalize()


class PermanenceVaryingSparsity(PermanenceFixedSparsity):
    """
    A weight matrix with permanence with varying sparsity learning rule.
    The binary weights sparsity vary throughout learning.
    """

    def __new__(cls, data, excitatory: bool,
                output_sparsity_desired=(0.025, 0.1)):
        """
        Convert a numpy array to the PermanenceVaryingSparsity instance.

        Parameters
        ----------
        data : (N_output, N_input) np.ndarray
            Initial binary connections weights.
        excitatory : bool
            Whether the connections are excitatory (True) or inhibitory (False)
        output_sparsity_desired : tuple of float
            The desired output sparsity range.
            Default: (0.025, 0.1)
        """
        if data is None:
            return None
        mat = super().__new__(cls, data)
        mat.excitatory = excitatory
        mat.output_sparsity_desired = output_sparsity_desired
        mat.s_w = np.random.random()  # target weight sparsity
        return mat

    def update_s_w(self, output_sparsity: float, gamma=0.1):
        """
        Update the s_w, the target weight sparsity.

        Parameters
        ----------
        output_sparsity : float
            The output population sparsity.
        gamma : float, optional
            The weight sparsity update speed.
            Default: 0.1

        Returns
        -------
        s_w : float
            A new value for s_w.
        """
        # 0.05 and 0.95 values are chosen arbitrary to prevent the saturation
        s_inc = min(0.95, self.s_w * (1 + gamma))
        s_dec = max(0.05, self.s_w * (1 - gamma))
        s_w = self.s_w
        s_min, s_max = self.output_sparsity_desired
        if output_sparsity > s_max:
            s_w = s_dec if self.excitatory else s_inc
        elif output_sparsity < s_min:
            s_w = s_inc if self.excitatory else s_dec
        return s_w

    def update(self, x_pre, x_post, n_choose=10, lr=0.001):
        """
        Update the connections weights from the pre- and post-synaptic
        activations.

        Parameters
        ----------
        x_pre, x_post : np.ndarray
            Pre- and post-synaptic activations. It's a 2D array, the first
            axis is neurons, and the second is the sample (trial) ID. The
            dimensionality of the fist axis can differ for the pre- and post-
            synaptic vectors.
        n_choose : int, optional
            Non-zero values to choose to update from the pre- and post- outer
            products.
            Default: 10
        lr : float, optional
            The learning rate.
            Default: 0.001
        """
        output_sparsity = np.count_nonzero(x_post) / x_post.size
        self.s_w = self.update_s_w(output_sparsity)
        super().update(x_pre, x_post, n_choose=n_choose, lr=lr)

    def normalize(self):
        """
        Normalize the permanence and binary weights matrices.
        """
        normalize_presynaptic(self.permanence)
        # Each output neuron will have 'k' synapses to input neurons.
        # The weight matrix shape is (N_out, N_in).
        k = math.ceil(self.s_w * self.shape[1])

        # Leave the 'k' largest entries of 'P' in 'w' for each output neuron
        winners = np.argsort(self.permanence, axis=1)[:, -k:]  # (N_out, k)
        self.fill(0)
        self[np.arange(self.shape[0])[:, np.newaxis], winners] = 1

        self.permanence *= self  # prune permanences, removed in weights
