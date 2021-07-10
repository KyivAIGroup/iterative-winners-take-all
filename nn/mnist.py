import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose

from mighty.loss import TripletLossSampler
from mighty.trainer import TrainerEmbedding, TrainerGrad
from mighty.utils.common import set_seed
from mighty.utils.data import DataLoader
from mighty.utils.domain import MonitorLevel
from mighty.utils.var_online import MeanOnline
from nn.kwta import *
from nn.nn_utils import l0_sparsity, sample_bernoulli
from nn.trainer import TrainerIWTA

set_seed(0)

N_x = 28 ** 2
N_h = N_y = 1024
s_x = 0.1
s_w_xh = s_w_xy = s_w_hy = s_w_yy = s_w_hh = s_w_yh = 0.05


class TrainerIWTAMnist(TrainerIWTA):
    N_CHOOSE = 100

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['sparsity-h'] = MeanOnline()
        return online

    def _epoch_finished(self, loss):
        self.monitor.update_kwta_thresholds(self.model.kwta_thresholds())
        self.monitor.update_weight_sparsity(self.model.weight_sparsity())
        self.monitor.update_weight_nonzero_keep(self.model.weight_nonzero_keep())
        self.monitor.update_sparsity(self.online['sparsity'].get_mean(),
                                     mode='y')
        self.monitor.update_sparsity(self.online['sparsity-h'].get_mean(),
                                     mode='h')
        mean, std = self.online['clusters'].get_mean_std()
        self.monitor.clusters_heatmap(mean)
        self.monitor.update_pairwise_dist(mean, std)
        TrainerGrad._epoch_finished(self, loss)

    def _on_forward_pass_batch(self, batch, output, train):
        h, y = output
        if train:
            sparsity_h = torch.Tensor([l0_sparsity(h)])
            self.online['sparsity-h'].update(sparsity_h)
        TrainerEmbedding._on_forward_pass_batch(self, batch, y, train)


w_xy = PermanenceVaryingSparsity(sample_bernoulli((N_x, N_y), p=s_w_xy), learn=False)
w_xh = PermanenceVaryingSparsity(sample_bernoulli((N_x, N_h), p=s_w_xh), learn=False)
w_hy = PermanenceVaryingSparsity(sample_bernoulli((N_h, N_y), p=s_w_hy), learn=True)
w_hh = PermanenceVaryingSparsity(sample_bernoulli((N_h, N_h), p=s_w_hy), learn=True)
w_yy = PermanenceVaryingSparsity(sample_bernoulli((N_y, N_y), p=s_w_yy), learn=True)
w_yh = PermanenceVaryingSparsity(sample_bernoulli((N_y, N_h), p=s_w_yh), learn=True)


class BinarizeMnist(nn.Module):
    def forward(self, tensor: torch.Tensor):
        tensor = (tensor > 0).float()
        return tensor


data_loader = DataLoader(MNIST, transform=Compose([ToTensor(), BinarizeMnist()]))
criterion = TripletLossSampler(nn.TripletMarginLoss())
iwta = IterativeWTA(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy, w_yh=w_yh)
# iwta = KWTANet(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, kh=10, ky=10)
print(iwta)

trainer = TrainerIWTAMnist(model=iwta, criterion=criterion,
                           data_loader=data_loader, verbosity=2)
trainer.monitor.advanced_monitoring(
    level=MonitorLevel.SIGN_FLIPS | MonitorLevel.WEIGHT_HISTOGRAM)
trainer.train(n_epochs=10)
