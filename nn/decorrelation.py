"""
Let y1 = f(x1, W) and y2 = f(x2, W).
Learning the weights either for (x1, y1) or (x2, y2) should decorrelate y1 and y2 signals.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn

from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset
from mighty.monitor.monitor import MonitorEmbedding
from constants import RESULTS_DIR

# from kwta import kWTA, iWTA, update_weights, overlap, RESULTS_DIR, kWTA_different_k
from nn.kwta import KWTANet, IterativeWTA, update_weights
from nn.utils import sample_bernoulli, NoShuffleLoader


from mighty.models import *
from mighty.monitor.accuracy import AccuracyArgmax, AccuracyEmbedding
from mighty.monitor.monitor import Monitor
from mighty.monitor.mutual_info import *
from mighty.trainer import *
from mighty.utils.common import set_seed
from mighty.utils.data import get_normalize_inverse, DataLoader, \
    TransformDefault
from mighty.utils.domain import MonitorLevel
from mighty.loss import TripletLossSampler

from nn.trainer import TrainerIWTA


N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh, s_w_yy = 0.5, 0.1, 0.1, 0.1, 0.05, 0.05
N_REPEATS, N_ITERS = 10, 100
K_FIXED = int(0.15 * N_y)
NUM_TO_LEARN = 5

stats = {mode: torch.zeros((N_REPEATS, N_ITERS), dtype=torch.float32)
         for mode in ('kWTA-fixed-k', 'kWTA', 'iWTA', 'nonzero')}


def overlap2d(tensor):
    assert tensor.shape[0] == 2
    return (tensor[0] & tensor[1]).sum()


class RandomDataset(TensorDataset):
    def __init__(self, *args, **kwargs):
        x12 = sample_bernoulli(s_x, shape=(2, N_x))
        labels = torch.arange(2)
        super().__init__(x12, labels)


for repeat in trange(N_REPEATS):
    w_xy, w_xh, w_hy, w_hh, w_yy = {}, {}, {}, {}, {}
    for mode in stats.keys():
        w_xy[mode] = sample_bernoulli(s_w_xy, shape=(N_x, N_y))
        w_xh[mode] = sample_bernoulli(s_w_xh, shape=(N_x, N_h))
        w_hy[mode] = sample_bernoulli(s_w_hy, shape=(N_h, N_y))
    kwta_variable = KWTANet(w_xy=w_xy['kWTA'], w_xh=w_xh['kWTA'], w_hy=w_hy['kWTA'], kh=K_FIXED)
    kwta_fixed = KWTANet(w_xy=w_xy['kWTA-fixed-k'], w_xh=w_xh['kWTA-fixed-k'], w_hy=w_hy['kWTA-fixed-k'], kh=K_FIXED, ky=K_FIXED)
    iwta = IterativeWTA(w_xy=w_xy['iWTA'], w_xh=w_xh['iWTA'], w_hy=w_hy['iWTA'])

    data_loader = DataLoader(RandomDataset, transform=None,
                             loader_cls=NoShuffleLoader)
    criterion = TripletLossSampler(nn.TripletMarginLoss())
    trainer = TrainerIWTA(model=iwta, criterion=criterion,
                          data_loader=data_loader)
    trainer.train(n_epochs=N_ITERS)

    x12, labels_unused = data_loader.sample()

    for iter_id in range(N_ITERS):
        h, y = {}, {}
        h['iWTA'], y['iWTA'] = iwta(x12)
        ky = torch.count_nonzero(y['iWTA'], dim=1)
        stats['nonzero'][repeat, iter_id] = ky.float().mean()

        h['kWTA'], y['kWTA'] = kwta_variable(x12, ky=ky)
        h['kWTA-fixed-k'], y['kWTA-fixed-k'] = kwta_fixed(x12)

        for mode in ('kWTA-fixed-k', 'kWTA', 'iWTA'):
            stats[mode][repeat, iter_id] = overlap2d(y[mode])
            update_weights(w_hy[mode], x_pre=h[mode], x_post=y[mode], n_choose=NUM_TO_LEARN)

colormap = {
    'iWTA': 'green',
    'kWTA': 'blue',
    'kWTA-fixed-k': 'cyan'
}

fig, ax = plt.subplots()
ax.plot(range(N_ITERS), stats.pop('nonzero').mean(axis=0), lw=2, ls='dashed',
        label='nonzero', color='gray')

for key in stats.keys():
    mean = stats[key].mean(axis=0)
    std = stats[key].std(axis=0)
    ax.plot(range(N_ITERS), mean, lw=2, label=key, color=colormap[key])
    ax.fill_between(range(N_ITERS), mean + std, mean - std,
                    facecolor=colormap[key], alpha=0.3)
ax.set_title("Decorrelation")
ax.legend()
ax.set_xlabel('Iteration')
ax.set_ylabel('Overlap($y_1$, $y_2$)')
plt.savefig(RESULTS_DIR / "decorrelation_nn.jpg")
# plt.show()
