import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from mighty.loss import ContrastiveLossSampler
from mighty.utils.common import set_seed
from mighty.utils.data import DataLoader
from nn.kwta import *
from nn.trainer import TrainerIWTA
from nn.utils import NoShuffleLoader

set_seed(0)

N_x = N_y = N_h = 200
s_x = 0.1
s_w_xh = 0.05
s_w_xy = 0.05
s_w_hy = 0.1
s_w_yy = 0.01
s_w_hh = 0.1
s_w_yh = 0.05

N_CLASSES = 2
N_SAMPLES_PER_CLASS = 10


class TrainerIWTAClustering(TrainerIWTA):
    pass


class NoisyCentroids(TensorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(xs, labels)


centroids = np.random.binomial(1, s_x, size=(N_x, N_CLASSES))
assert centroids.any(axis=0).all(), "Pick another seed"

xs = np.repeat(centroids, repeats=N_SAMPLES_PER_CLASS, axis=1)
labels = np.repeat(np.arange(N_CLASSES), N_SAMPLES_PER_CLASS)
white_noise = np.random.binomial(1, 0.1, size=xs.shape)
xs ^= white_noise

xs = torch.from_numpy(xs.T).type(torch.int32)
labels = torch.from_numpy(labels)

w_xy = ParameterWithPermanence(torch.rand(N_x, N_y), sparsity=s_w_xy, learn=False)
w_xh = ParameterWithPermanence(torch.rand(N_x, N_h), sparsity=s_w_xh, learn=False)
w_hy = ParameterWithPermanence(torch.rand(N_h, N_y), sparsity=s_w_hy)
w_hh = ParameterWithPermanence(torch.rand(N_h, N_h), sparsity=s_w_hh, learn=False)
w_yh = ParameterWithPermanence(torch.rand(N_y, N_h), sparsity=s_w_yh, learn=False)
w_yy = ParameterWithPermanence(torch.rand(N_y, N_y), sparsity=s_w_yy)

iwta = IterativeWTA(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy, w_yh=w_yh)
print(iwta)

data_loader = DataLoader(NoisyCentroids, transform=None,
                         loader_cls=NoShuffleLoader)
criterion = ContrastiveLossSampler(nn.CosineEmbeddingLoss(margin=0),
                                   pairs_multiplier=5)
trainer = TrainerIWTAClustering(model=iwta, criterion=criterion,
                                data_loader=data_loader, verbosity=1)
# trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
iwta.set_monitor(trainer.monitor)
trainer.train(n_epochs=10)
