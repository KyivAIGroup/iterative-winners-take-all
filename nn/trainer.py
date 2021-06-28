import torch
import torch.nn as nn

from mighty.monitor.accuracy import AccuracyEmbedding
from mighty.trainer import TrainerEmbedding, TrainerGrad
from mighty.utils.common import clone_cpu
from mighty.utils.data import DataLoader
from mighty.utils.signal import compute_sparsity
from mighty.utils.stub import OptimizerStub
from mighty.utils.var_online import MeanOnline
from nn.kwta import WTAInterface, IterativeWTASoft
from nn.monitor import MonitorIWTA
from utils import compute_discriminative_factor


class TrainerIWTA(TrainerEmbedding):

    watch_modules = TrainerEmbedding.watch_modules + (WTAInterface,)
    N_CHOOSE = 1

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 optimizer=OptimizerStub(),
                 **kwargs):
        super().__init__(model=model,
                         criterion=criterion,
                         data_loader=data_loader,
                         optimizer=optimizer,
                         accuracy_measure=AccuracyEmbedding(cache=True),
                         **kwargs)
        self.mutual_info.save_activations = self.mi_save_activations_y
        self.cached_labels = []
        self.cached_output = []

    def mi_save_activations_y(self, module, tin, tout):
        """
        A hook to save the activates at a forward pass.
        """
        if not self.mutual_info.is_updating:
            return
        h, y = tout
        layer_name = self.mutual_info.layer_to_name[module]
        tout_clone = clone_cpu(y.detach().float())
        tout_clone = tout_clone.flatten(start_dim=1)
        self.mutual_info.activations[layer_name].append(tout_clone)

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
        loss = self._get_loss(batch, (h, y))
        if isinstance(self.model, IterativeWTASoft):
            loss.backward()
            self.optimizer.step(closure=None)
        else:
            self.model.update_weights(x, h, y, n_choose=self.N_CHOOSE)
        return loss

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['sparsity-h'] = MeanOnline()  # scalar
        return online

    def training_started(self):
        x, labels = self.data_loader.sample()
        h, y = self.model(x)
        self.monitor.update_pairwise_similarity(x, labels, name='x')
        self.monitor.update_pairwise_similarity(y, labels, name='y')

    def update_discriminative_factor(self):
        if len(self.cached_labels) == 0:
            return
        labels_true = torch.cat(self.cached_labels).numpy()
        output = torch.cat(self.cached_output).numpy()
        factor = compute_discriminative_factor(output, labels_true)
        self.monitor.update_discriminative_factor(factor)
        self.cached_output.clear()
        self.cached_labels.clear()

    def _epoch_finished(self, loss):
        x, labels = self.data_loader.sample()
        self.monitor.track_iwta = self.timer.epoch in (1, self.timer.n_epochs)
        h, y = self.model(x)
        self.monitor.track_iwta = False
        self.monitor.plot_assemblies(h, labels, name='h')
        self.monitor.plot_assemblies(y, labels, name='y')
        self.monitor.update_pairwise_similarity(y, labels, name='y')
        self.monitor.update_weight_sparsity(self.model.weight_sparsity())
        self.update_discriminative_factor()

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
            x, labels = batch
            self.cached_labels.append(labels)
            self.cached_output.append(y)
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
