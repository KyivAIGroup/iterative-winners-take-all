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
from nn.kwta import *
from nn.trainer import TrainerIWTA
from nn.utils import sample_bernoulli, NoShuffleLoader

# N_x, N_y, N_h = 100, 100, 100
N_x = N_y = N_h = 200
s_x = 0.1
s_w_xh = 0.05
s_w_xy = 0.05
s_w_hy = 0.1

WITH_PERMANENCE = True
N_REPEATS, N_ITERS = 10, 50
K_FIXED = int(0.15 * N_y)
NUM_TO_LEARN = 50

stats = {mode: torch.zeros((N_REPEATS, N_ITERS), dtype=torch.float32)
         for mode in ('kWTA-fixed-k', 'kWTA', 'iWTA', 'nonzero')}


class TrainerIWTADecorrelation(TrainerIWTA):
    def train_batch(self, batch):
        return super().train_batch(batch)
        x, labels = batch
        h, y = self.model(x)
        loss = self._get_loss(batch, (h, y))
        if isinstance(self.model.w_hy, ParameterWithPermanence):
            self.model.w_hy.update(x_pre=h, x_post=y)
        else:
            update_weights(self.model.w_hy, x_pre=h, x_post=y, n_choose=NUM_TO_LEARN)
        return loss


class RandomDataset(TensorDataset):
    def __init__(self, *args, **kwargs):
        labels = torch.arange(2)
        super().__init__(x12, labels)

x12 = sample_bernoulli((2, N_x), p=s_x)
w_xy, w_xh, w_hy, w_hh, w_yy = {}, {}, {}, {}, {}
for mode in stats.keys():
    if WITH_PERMANENCE:
        w_xy[mode] = ParameterWithPermanence.generate_sparse((N_x, N_y), sparsity=s_w_xy)
        w_xh[mode] = ParameterWithPermanence.generate_sparse((N_x, N_h), sparsity=s_w_xh)
        w_hy[mode] = ParameterWithPermanence.generate_sparse((N_h, N_y), sparsity=s_w_hy)
    else:
        w_xy[mode] = nn.Parameter(sample_bernoulli((N_x, N_y), p=s_w_xy), requires_grad=False)
        w_xh[mode] = nn.Parameter(sample_bernoulli((N_x, N_h), p=s_w_xh), requires_grad=False)
        w_hy[mode] = nn.Parameter(sample_bernoulli((N_h, N_y), p=s_w_hy), requires_grad=False)
kwta_variable = KWTANet(w_xy=w_xy['kWTA'], w_xh=w_xh['kWTA'], w_hy=w_hy['kWTA'], kh=K_FIXED)
kwta_fixed = KWTANet(w_xy=w_xy['kWTA-fixed-k'], w_xh=w_xh['kWTA-fixed-k'], w_hy=w_hy['kWTA-fixed-k'], kh=K_FIXED, ky=K_FIXED)
iwta = IterativeWTASparse(w_xy=w_xy['iWTA'], w_xh=w_xh['iWTA'], w_hy=w_hy['iWTA'])

data_loader = DataLoader(RandomDataset, transform=None,
                         loader_cls=NoShuffleLoader)
criterion = ContrastiveLossSampler(nn.CosineEmbeddingLoss(margin=0),
                                   pairs_multiplier=5)
trainer = TrainerIWTADecorrelation(model=iwta, criterion=criterion,
                                   data_loader=data_loader, verbosity=1)
# trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
iwta.set_monitor(trainer.monitor)
trainer.train(n_epochs=10)
