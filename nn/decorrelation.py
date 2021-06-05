"""
Let y1 = f(x1, W) and y2 = f(x2, W).
Learning the weights either for (x1, y1) or (x2, y2) should decorrelate y1 and y2 signals.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from tqdm import trange

from constants import RESULTS_DIR
from mighty.loss import ContrastiveLossSampler
from mighty.utils.data import DataLoader
from mighty.utils.domain import MonitorLevel
from nn.kwta import KWTANet, IterativeWTA, update_weights
from nn.trainer import TrainerIWTA
from nn.utils import sample_bernoulli, NoShuffleLoader

N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh, s_w_yy = 0.5, 0.1, 0.1, 0.1, 0.05, 0.05
N_REPEATS, N_ITERS = 10, 50
K_FIXED = int(0.15 * N_y)
NUM_TO_LEARN = 5

stats = {mode: torch.zeros((N_REPEATS, N_ITERS), dtype=torch.float32)
         for mode in ('kWTA-fixed-k', 'kWTA', 'iWTA', 'nonzero')}


class TrainerIWTADecorrelation(TrainerIWTA):
    def train_batch(self, batch):
        h, y = self.model(batch[0])
        loss = self._get_loss(batch, (h, y))
        update_weights(self.model.w_hy, x_pre=h, x_post=y, n_choose=5)
        return loss


class RandomDataset(TensorDataset):
    def __init__(self, *args, **kwargs):
        labels = torch.arange(2)
        super().__init__(x12, labels)

x12 = sample_bernoulli(s_x, shape=(2, N_x))
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
criterion = ContrastiveLossSampler(nn.CosineEmbeddingLoss(margin=0),
                                   pairs_multiplier=5)
trainer = TrainerIWTADecorrelation(model=iwta, criterion=criterion,
                                   data_loader=data_loader, verbosity=1)
trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
trainer.train(n_epochs=N_ITERS)
