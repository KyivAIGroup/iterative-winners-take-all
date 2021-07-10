"""
Let y1 = f(x1, W) and y2 = f(x2, W).
Learning the weights either for (x1, y1) or (x2, y2) should decorrelate y1 and y2 signals.
"""

import numpy as np
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
s_x = 0.1
s_w_xh = s_w_xy = s_w_hy = s_w_yy = s_w_hh = s_w_yh = 0.05

N_UNIQUE = 10
N_REPEATS = 100
N_REPEATS_PEAK = 40


class TrainerIWTAHabituation(TrainerIWTA):
    N_CHOOSE = None

    def _update_cached(self):
        labels = torch.cat(self.cached_labels)
        labels_unique = labels.unique().tolist()
        convergence = {}
        sparsity_per_label = {"p(x)": p_x}
        for name, output in self.cached_output.items():
            output = torch.cat(output).int()
            sparsity = torch.zeros(len(labels_unique))
            if name in self.cached_output_prev:
                xor = (self.cached_output_prev[name] ^ output).sum(dim=1)
                convergence[name] = xor.float().mean().item() / output.size(1)
            self.cached_output_prev[name] = output
            for label in labels_unique:
                mask = labels == label
                sparsity[label] = l0_sparsity(output[mask])
            sparsity /= sparsity.sum()
            sparsity_per_label[f"||{name}||_0"] = sparsity
        self.monitor.update_output_convergence(convergence)
        self.monitor.update_sparsity_per_label(sparsity_per_label)
        self.cached_output.clear()
        self.cached_labels.clear()


class RandomWithRepetitions(TensorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(xs, labels)


x_unique = sample_bernoulli((N_UNIQUE, N_x), p=s_x)
labels = np.random.choice(N_UNIQUE - 1, size=N_REPEATS - N_REPEATS_PEAK)
labels = np.r_[labels, np.full(N_REPEATS_PEAK, fill_value=N_UNIQUE - 1)]
np.random.shuffle(labels)
labels = torch.from_numpy(labels).to(device=x_unique.device)
xs = x_unique[labels]

_, label_counts = labels.unique(return_counts=True)
p_x = label_counts.cpu() / len(labels)

w_xy = PermanenceVaryingSparsity(sample_bernoulli((N_x, N_y), p=s_w_xy), learn=False)
w_xh = PermanenceVaryingSparsity(sample_bernoulli((N_x, N_h), p=s_w_xh), learn=False)
w_hy = PermanenceVaryingSparsity(sample_bernoulli((N_h, N_y), p=s_w_hy), learn=True)
w_hh = PermanenceVaryingSparsity(sample_bernoulli((N_h, N_h), p=s_w_hy), learn=True)
w_yy = PermanenceVaryingSparsity(sample_bernoulli((N_y, N_y), p=s_w_yy), learn=True)
w_yh = PermanenceVaryingSparsity(sample_bernoulli((N_y, N_h), p=s_w_yh), learn=True)
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
