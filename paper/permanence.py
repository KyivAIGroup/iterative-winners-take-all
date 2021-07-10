import math
import numpy as np

from kwta import kWTA


def normalize_presynaptic(mat):
    presum = mat.sum(axis=1)[:, np.newaxis]
    presum += 1e-10  # avoid division by zero
    mat /= presum


class PermanenceFixedSparsity(np.ndarray):

    def __new__(cls, data, **kwargs):
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

    @staticmethod
    def kWTA_threshold(vec, k):
        vec_nonzero = np.sort(vec[vec > 0])
        k = min(k, len(vec_nonzero))
        thr = vec_nonzero[-k]
        return thr

    def update(self, x_pre, x_post, n_choose=1, lr=0.001):
        assert x_pre.shape[1] == x_post.shape[1], "Batch size mismatch"
        for x, y in zip(x_pre.T, x_post.T):
            x = x.nonzero()[0]
            y = y.nonzero()[0]
            if len(x) == 0 or len(y) == 0:
                continue
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


class PermanenceVogels(PermanenceFixedSparsity):

    def update(self, x_pre, x_post, n_choose=1, lr=0.001,
               neighbors_coincident=1):
        assert len(x_pre) == len(x_post)
        window_size = (neighbors_coincident + 1)
        n_steps = len(x_pre)
        lr_potentiation = lr / (n_steps * window_size)
        lr_depression = -lr / (n_steps * (n_steps - window_size))
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


class PermanenceVaryingSparsity(PermanenceFixedSparsity):

    def __new__(cls, data, excitatory: bool,
                output_sparsity_desired=(0.025, 0.1)):
        mat = super().__new__(cls, data)
        mat.excitatory = excitatory
        mat.output_sparsity_desired = output_sparsity_desired
        mat.weight_nonzero_keep = np.random.random()
        return mat

    def update_weight_sparsity(self, output_sparsity: float, gamma=0.1):
        sparsity_inc = gamma * 0.95 + (1 - gamma) * self.weight_nonzero_keep
        sparsity_dec = gamma * 0.05 + (1 - gamma) * self.weight_nonzero_keep
        sparsity = self.weight_nonzero_keep
        s_min, s_max = self.output_sparsity_desired
        if output_sparsity > s_max:
            sparsity = sparsity_dec if self.excitatory else sparsity_inc
        elif output_sparsity < s_min:
            sparsity = sparsity_inc if self.excitatory else sparsity_dec
        return sparsity

    def update(self, x_pre, x_post, n_choose=1, lr=0.001):
        output_sparsity = np.count_nonzero(x_post) / x_post.size
        self.weight_nonzero_keep = self.update_weight_sparsity(output_sparsity)
        super().update(x_pre, x_post, n_choose=n_choose, lr=lr)

    def normalize(self):
        normalize_presynaptic(self.permanence)
        k = math.ceil(self.weight_nonzero_keep * self.size)
        threshold = self.kWTA_threshold(self.permanence.reshape(-1), k=k)
        self.permanence[self.permanence < threshold] = 0
        self[:] = self.permanence > 0
