"""
Let y1 = f(x1, W) and y2 = f(x2, W).
Learning the weights either for (x1, y1) or (x2, y2) should decorrelate y1 and y2 signals.
"""

import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from mighty.loss import ContrastiveLossSampler
from mighty.utils.common import set_seed
from mighty.utils.data import DataLoader
from mighty.utils.domain import MonitorLevel
from nn.kwta import *
from nn.nn_utils import sample_bernoulli, NoShuffleLoader, l0_sparsity
from nn.trainer import TrainerIWTA

set_seed(0)

N_x = N_y = N_h = 200
s_x = 0.2
s_w_xh = s_w_xy = s_w_hy = s_w_yy = s_w_hh = s_w_yh = 0.05

N_SAMPLES_TOTAL = 30


def generate_k_active(n, k):
    x = np.zeros(n, dtype=np.int32)
    active = np.random.choice(n, size=k, replace=False)
    x[active] = 1
    return x


def sample_from_distribution(px, n_neurons, n_samples, k):
    px = np.array(px)
    assert np.isclose(px.sum(), 1), "Probabilities must sum up to 1"
    x = np.array([generate_k_active(n_neurons, k) for pxi in px])
    labels = []
    for i, pxi in enumerate(px):
        repeats = math.ceil(pxi * n_samples)
        labels_repeated = np.full(repeats, fill_value=i)
        labels.append(labels_repeated)
    labels = np.hstack(labels)
    np.random.shuffle(labels)
    x = x[labels]
    x = torch.from_numpy(x).float()
    labels = torch.from_numpy(labels)
    if torch.cuda.is_available():
        x = x.cuda()
        labels = labels.cuda()
    return x, labels


class TrainerIWTAHabituation(TrainerIWTA):
    N_CHOOSE = 100
    LEARNING_RATE = 0.001

    def _update_cached(self):
        labels = torch.cat(self.cached_labels)
        labels_unique = labels.unique().tolist()
        convergence = {}
        sparsity = {}
        sparsity_per_label = {"p(x)": px}
        for name, output in self.cached_output.items():
            output = torch.cat(output).int()
            sparsity[name] = l0_sparsity(output)
            if name in self.cached_output_prev:
                xor = (self.cached_output_prev[name] ^ output).sum(dim=1)
                convergence[name] = xor.float().mean().item() / output.size(1)
            self.cached_output_prev[name] = output
            sparsity_per_label[f"s_{name}"] = [l0_sparsity(output[labels == l])
                                               for l in labels_unique]
        self.monitor.update_output_convergence(convergence)
        self.monitor.update_sparsity_per_label(sparsity_per_label)
        self.monitor.update_sparsity(sparsity)
        self.cached_output.clear()
        self.cached_labels.clear()


class RandomWithRepetitions(TensorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(x, labels)


px = [0.2, 0.2, 0.6]
x, labels = sample_from_distribution(px=px, n_neurons=N_x, n_samples=N_SAMPLES_TOTAL, k=int(s_x * N_x))

Permanence = PermanenceFixedSparsity

w_xy = Permanence(sample_bernoulli((N_x, N_y), p=s_w_xy), excitatory=True, learn=True)
w_xh = Permanence(sample_bernoulli((N_x, N_h), p=s_w_xh), excitatory=True, learn=True)
w_hy = Permanence(sample_bernoulli((N_h, N_y), p=s_w_hy), excitatory=False, learn=True)
w_hh = Permanence(sample_bernoulli((N_h, N_h), p=s_w_hy), excitatory=False, learn=True)
w_yy = None
w_yh = Permanence(sample_bernoulli((N_y, N_h), p=s_w_yh), excitatory=True, learn=True)

# w_hh = Permanence(sample_bernoulli((N_h, N_h), p=s_w_hy), learn=True)
# w_yy = None
# w_yh = None

iwta = IterativeWTA(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy, w_yh=w_yh)
# iwta = KWTANet(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, kh=10, ky=10)
print(iwta)

data_loader = DataLoader(RandomWithRepetitions, transform=None,
                         loader_cls=NoShuffleLoader)
criterion = ContrastiveLossSampler(nn.CosineEmbeddingLoss(margin=0))
trainer = TrainerIWTAHabituation(model=iwta, criterion=criterion,
                                   data_loader=data_loader, verbosity=1)
trainer.monitor.advanced_monitoring(level=MonitorLevel.SIGN_FLIPS | MonitorLevel.WEIGHT_HISTOGRAM)
trainer.train(n_epochs=50)
