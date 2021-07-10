import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from mighty.loss import ContrastiveLossSampler
from mighty.utils.common import set_seed
from mighty.utils.data import DataLoader
from mighty.utils.domain import MonitorLevel
from nn.kwta import *
from nn.nn_utils import NoShuffleLoader, sample_bernoulli
from nn.trainer import TrainerIWTA

set_seed(0)

N_x = N_y = N_h = 200
s_x = 0.2
s_w_xh = s_w_xy = s_w_hy = s_w_yy = s_w_hh = s_w_yh = 0.05
# s_w_hh = s_w_yy = 0.01
# s_w_hy = s_w_yh = 0.1
# s_w_xh = s_w_xy = 0.01


N_CLASSES = 10
N_SAMPLES_PER_CLASS = 100


class TrainerIWTAClustering(TrainerIWTA):
    N_CHOOSE = 100
    pass


class NoisyCentroids(TensorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(x, labels)


centroids = np.random.binomial(1, s_x, size=(N_CLASSES, N_x))
assert centroids.any(axis=1).all(), "Pick another seed"
labels = np.repeat(np.arange(N_CLASSES), N_SAMPLES_PER_CLASS)
np.random.shuffle(labels)
x = centroids[labels]
white_noise = np.random.binomial(1, 0.5 * s_x, size=x.shape)
x ^= white_noise

x = torch.from_numpy(x).float()
labels = torch.from_numpy(labels)
if torch.cuda.is_available():
    x = x.cuda()
    labels = labels.cuda()

Permanence = PermanenceVaryingSparsity

w_xy = Permanence(sample_bernoulli((N_x, N_y), p=s_w_xy), learn=False)
w_xh = Permanence(sample_bernoulli((N_x, N_h), p=s_w_xh), learn=False)
w_hy = Permanence(sample_bernoulli((N_h, N_y), p=s_w_hy), learn=True)
w_hh = Permanence(sample_bernoulli((N_h, N_h), p=s_w_hy), learn=True)
w_yy = Permanence(sample_bernoulli((N_y, N_y), p=s_w_yy), learn=True)
w_yh = Permanence(sample_bernoulli((N_y, N_h), p=s_w_yh), learn=True)
# w_hh = None
# w_yy = None
# w_yh = None

data_loader = DataLoader(NoisyCentroids, transform=None,
                         loader_cls=NoShuffleLoader)
criterion = ContrastiveLossSampler(nn.CosineEmbeddingLoss(margin=0))

# iwta = IterativeWTA(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy, w_yh=w_yh)
iwta = KWTANet(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, kh=10, ky=10)
print(iwta)

trainer = TrainerIWTAClustering(model=iwta, criterion=criterion,
                                   data_loader=data_loader, verbosity=1)
trainer.monitor.advanced_monitoring(level=MonitorLevel.SIGN_FLIPS | MonitorLevel.WEIGHT_HISTOGRAM)
trainer.train(n_epochs=20)
