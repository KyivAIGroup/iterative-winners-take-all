import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize, InterpolationMode

from mighty.loss import TripletLossSampler
from mighty.trainer import TrainerEmbedding, TrainerGrad
from mighty.utils.common import set_seed
from mighty.utils.data import DataLoader
from mighty.utils.domain import MonitorLevel
from mighty.utils.var_online import MeanOnline, MeanOnlineLabels
from nn.kwta import *
from nn.nn_utils import l0_sparsity, sample_bernoulli
from nn.trainer import TrainerIWTA

set_seed(0)

# N_x = 28 ** 2
# N_h = N_y = 1024
size = 28
N_x = N_h = N_y = size ** 2
s_w_xh = s_w_xy = s_w_hy = s_w_yy = s_w_hh = s_w_yh = 0.05
K_FIXED = int(0.05 * N_y)


class TrainerIWTAMnist(TrainerIWTA):
    N_CHOOSE = 10
    LEARNING_RATE = 0.01

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['sparsity-h'] = MeanOnline()
        online['clusters-h'] = MeanOnlineLabels()
        return online

    def _epoch_finished(self, loss):
        self.monitor.param_records.plot_sign_flips(self.monitor.viz)
        self.monitor.update_contribution(self.model.weight_contribution())
        self.monitor.update_kwta_thresholds(self.model.kwta_thresholds())
        self.monitor.update_weight_sparsity(self.model.weight_sparsity())
        self.monitor.update_weight_nonzero_keep(self.model.weight_nonzero_keep())
        self.monitor.update_sparsity(self.online['sparsity'].get_mean().item(), mode='y')
        self.monitor.update_sparsity(self.online['sparsity-h'].get_mean().item(), mode='h')
        self.monitor.clusters_heatmap(self.online['clusters'].get_mean(), title="Embeddings 'y'")
        self.monitor.clusters_heatmap(self.online['clusters-h'].get_mean(), title="Embeddings 'h'")
        TrainerGrad._epoch_finished(self, loss)

    def train_batch(self, batch):
        def centroids(tensor):
            return torch.stack([tensor[labels == l].mean(dim=0)
                                for l in labels.unique()])

        x, labels = batch
        h, y = self.model(x)
        self.update_contribution(h, y)
        loss = self._get_loss(batch, (h, y))
        self.model.update_weights(x, h, y, n_choose=self.N_CHOOSE,
                                  lr=self.LEARNING_RATE)

        if self.timer.epoch == 0:
            self.monitor.viz.images(x[:10], nrow=5, win="samples", opts=dict(
                width=500,
            ))
            self.online['clusters'].update(y, labels)
            self.online['clusters-h'].update(h, labels)
            self.online['sparsity'].update(torch.Tensor([l0_sparsity(y)]))
            self.online['sparsity-h'].update(torch.Tensor([l0_sparsity(h)]))
            if self.timer.batch_id > 0:
                self.monitor.param_records.plot_sign_flips(self.monitor.viz)
            self.monitor.update_weight_histogram()
            self.monitor.update_contribution(self.model.weight_contribution())
            self.monitor.update_kwta_thresholds(self.model.kwta_thresholds())
            self.monitor.update_weight_sparsity(self.model.weight_sparsity())
            self.monitor.update_weight_nonzero_keep(self.model.weight_nonzero_keep())
            self.monitor.update_sparsity(self.online['sparsity'].get_mean().item(), mode='y')
            self.monitor.update_sparsity(self.online['sparsity-h'].get_mean().item(), mode='h')
            self.monitor.clusters_heatmap(centroids(y), title="Embeddings 'y'")
            self.monitor.clusters_heatmap(centroids(h), title="Embeddings 'h'")

        return loss

    def training_started(self):
        self.monitor.update_weight_sparsity(self.model.weight_sparsity())
        self.monitor.update_weight_nonzero_keep(self.model.weight_nonzero_keep())

    def _on_forward_pass_batch(self, batch, output, train):
        h, y = output
        input, labels = batch
        if train:
            self.online['sparsity'].update(torch.Tensor([l0_sparsity(y)]))
            self.online['sparsity-h'].update(torch.Tensor([l0_sparsity(h)]))
            self.online['clusters'].update(y, labels)
            self.online['clusters-h'].update(h, labels)
        TrainerGrad._on_forward_pass_batch(self, batch, y, train)


Permanence = PermanenceVaryingSparsity

w_xy = Permanence(sample_bernoulli((N_x, N_y), p=s_w_xy), excitatory=True, learn=True)
w_xh = Permanence(sample_bernoulli((N_x, N_h), p=s_w_xh), excitatory=True, learn=True)
w_hy = Permanence(sample_bernoulli((N_h, N_y), p=s_w_hy), excitatory=False, learn=True)
w_hh = Permanence(sample_bernoulli((N_h, N_h), p=s_w_hy), excitatory=False, learn=True)
w_yh = Permanence(sample_bernoulli((N_y, N_h), p=s_w_yh), excitatory=True, learn=True)
w_yy = None


class BinarizeMnist(nn.Module):
    def forward(self, tensor: torch.Tensor):
        bern = torch.distributions.bernoulli.Bernoulli(0.05)
        noise = bern.sample(tensor.shape).bool()
        tensor = tensor > 0
        tensor ^= noise
        return tensor.float()


transform = [ToTensor(), BinarizeMnist()]
if size != 28:
    transform.insert(0, Resize(size, interpolation=InterpolationMode.NEAREST))
data_loader = DataLoader(MNIST, transform=Compose(transform), batch_size=256)

criterion = TripletLossSampler(nn.TripletMarginLoss())
iwta = IterativeWTA(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy, w_yh=w_yh)
# iwta = KWTANet(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, kh=K_FIXED, ky=K_FIXED)
print(iwta)

trainer = TrainerIWTAMnist(model=iwta, criterion=criterion,
                           data_loader=data_loader, verbosity=2)
trainer.monitor.advanced_monitoring(
    level=MonitorLevel.SIGN_FLIPS | MonitorLevel.WEIGHT_HISTOGRAM)
trainer.train(n_epochs=10)
