import torch.nn as nn

from mighty.trainer import TrainerEmbedding, TrainerGrad
from mighty.utils.data import DataLoader
from mighty.utils.stub import OptimizerStub
from mighty.monitor.accuracy import AccuracyEmbedding
from mighty.utils.var_online import MeanOnline
from mighty.utils.signal import compute_sparsity

from nn.monitor import MonitorIWTA
from nn.kwta import WTAInterface


class TrainerIWTA(TrainerEmbedding):

    watch_modules = TrainerEmbedding.watch_modules + (WTAInterface,)

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 env_suffix='',
                 **kwargs):
        super().__init__(model=model,
                         criterion=criterion,
                         data_loader=data_loader,
                         optimizer=OptimizerStub(),
                         scheduler=None,
                         accuracy_measure=AccuracyEmbedding(cache=True),
                         env_suffix=env_suffix,
                         **kwargs)

    def _init_monitor(self, mutual_info):
        monitor = MonitorIWTA(
            mutual_info=mutual_info,
            normalize_inverse=self.data_loader.normalize_inverse
        )
        return monitor

    def full_forward_pass(self, train=True):
        if not train:
            return None
        return super().full_forward_pass(train=train)

    def _forward(self, batch):
        h, y = super()._forward(batch)
        return h, y

    def train_batch(self, batch):
        x, labels = batch
        h, y = self.model(x)
        self.model.update_weights(x, h, y)
        loss = self._get_loss(batch, (h, y))
        return loss

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['sparsity-h'] = MeanOnline()  # scalar
        return online

    def _epoch_finished(self, loss):
        x, labels = self.data_loader.sample()
        self.monitor.track_iwta = self.timer.epoch in (1, self.timer.n_epochs)
        h, y = self.model(x)
        self.monitor.track_iwta = False
        self.monitor.plot_assemblies(h, name='h')
        self.monitor.plot_assemblies(y, name='y')
        self.monitor.update_weight_sparsity(self.model.weight_sparsity())

        self.monitor.update_sparsity(self.online['sparsity'].get_mean(),
                                     mode='y')
        self.monitor.update_sparsity(self.online['sparsity-h'].get_mean(),
                                     mode='h')
        self.monitor.update_l1_neuron_norm(self.online['l1_norm'].get_mean())
        # mean and std can be Nones
        mean, std = self.online['clusters'].get_mean_std()
        self.monitor.clusters_heatmap(mean=mean, std=std)
        self.monitor.embedding_hist(activations=mean)
        TrainerGrad._epoch_finished(self, loss)

    def _on_forward_pass_batch(self, batch, output, train):
        h, y = output
        if train:
            sparsity_h = compute_sparsity(h.float())
            self.online['sparsity-h'].update(sparsity_h.cpu())
        super()._on_forward_pass_batch(batch, y, train)

    def _get_loss(self, batch, output):
        # In case of unsupervised learning, '_get_loss' is overridden
        # accordingly.
        input, labels = batch
        h, y = output
        return self.criterion(y, labels)

    def training_finished(self):
        self.monitor._plot_iwta_heatmap()
