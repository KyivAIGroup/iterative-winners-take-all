import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from mighty.loss import ContrastiveLossSampler
from mighty.utils.common import set_seed
from mighty.utils.data import DataLoader
from nn.kwta import *
from nn.trainer import TrainerIWTA
from nn.nn_utils import NoShuffleLoader, sample_bernoulli
from mighty.utils.domain import MonitorLevel

set_seed(0)

N_x = N_y = N_h = 200
s_x = 0.1
s_w_xh = 0.1
s_w_xy = 0.1
s_w_hy = 0.1
s_w_yy = 0.1
s_w_hh = 0.1
s_w_yh = 0.1

N_CLASSES = 10
N_SAMPLES_PER_CLASS = 100


class TrainerIWTAClustering(TrainerIWTA):
    N_CHOOSE = 100
    pass


class NoisyCentroids(TensorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(xs, labels)


centroids = np.random.binomial(1, s_x, size=(N_x, N_CLASSES))
assert centroids.any(axis=0).all(), "Pick another seed"

xs = np.repeat(centroids, repeats=N_SAMPLES_PER_CLASS, axis=1).T
labels = np.repeat(np.arange(N_CLASSES), N_SAMPLES_PER_CLASS)
white_noise = np.random.binomial(1, 0.2 * s_x, size=xs.shape)
xs ^= white_noise
shuffle_idx = np.random.permutation(len(xs))
xs = xs[shuffle_idx]
labels = labels[shuffle_idx]

xs = torch.from_numpy(xs).float()
labels = torch.from_numpy(labels)
if torch.cuda.is_available():
    xs = xs.cuda()
    labels = labels.cuda()

w_xy = ParameterBinary(sample_bernoulli((N_x, N_y), p=s_w_xy), learn=False)
w_xh = ParameterBinary(sample_bernoulli((N_x, N_h), p=s_w_xh), learn=False)
w_hy = ParameterBinary(sample_bernoulli((N_h, N_y), p=s_w_hy), learn=True, dropout=0.5)
w_hh = ParameterBinary(sample_bernoulli((N_h, N_h), p=s_w_hy), learn=True, dropout=0.5)
w_yy = ParameterBinary(sample_bernoulli((N_y, N_y), p=s_w_yy), learn=True, dropout=0.5)
# w_yy = ParameterWithPermanence(torch.rand(N_y, N_y), sparsity=s_w_yy, learn=True)
w_yh = ParameterBinary(sample_bernoulli((N_y, N_h), p=s_w_yh), learn=True, dropout=0.5)
# w_yh = None

data_loader = DataLoader(NoisyCentroids, transform=None,
                         loader_cls=NoShuffleLoader)
criterion = ContrastiveLossSampler(nn.CosineEmbeddingLoss(margin=0))

iwta = IterativeWTA(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy, w_yh=w_yh)
# iwta = KWTANet(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, kh=10, ky=10)
print(iwta)

trainer = TrainerIWTAClustering(model=iwta, criterion=criterion,
                                   data_loader=data_loader, verbosity=1)
trainer.monitor.advanced_monitoring(level=MonitorLevel.SIGN_FLIPS | MonitorLevel.WEIGHT_HISTOGRAM)
trainer.train(n_epochs=50)
