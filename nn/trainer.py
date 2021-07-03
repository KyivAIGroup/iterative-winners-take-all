from collections import defaultdict

import torch
import torch.nn as nn

from mighty.monitor.accuracy import AccuracyEmbedding
from mighty.trainer import TrainerEmbedding, TrainerGrad
from mighty.utils.common import clone_cpu
from mighty.utils.data import DataLoader
from mighty.utils.stub import OptimizerStub
from nn.kwta import WTAInterface, IterativeWTASoft
from nn.monitor import MonitorIWTA
from nn.nn_utils import compute_discriminative_factor, l0_sparsity


class TrainerIWTA(TrainerEmbedding):

    watch_modules = TrainerEmbedding.watch_modules + (WTAInterface,)
    N_CHOOSE = None
    LEARNING_RATE = 0.001

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
        self.cached_output = defaultdict(list)
        self.cached_output_prev = {}

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

    def train_batch(self, batch):
        x, labels = batch
        h, y = self.model(x)
        loss = self._get_loss(batch, (h, y))
        if isinstance(self.model, IterativeWTASoft):
            loss.backward()
            self.optimizer.step(closure=None)
        else:
            self.model.update_weights(x, h, y, n_choose=self.N_CHOOSE,
                                      lr=self.LEARNING_RATE)
        return loss

    def _update_cached(self):
        labels = torch.cat(self.cached_labels)
        factors = {}
        convergence = {}
        sparsity = {}
        for name, output in self.cached_output.items():
            output = torch.cat(output)
            if name == 'y':
                mean = [output[labels == label].mean(dim=0)
                        for label in labels.unique()]
                mean = torch.stack(mean)
                self.monitor.clusters_heatmap(mean)
            # self.monitor.plot_assemblies(output, labels, name=name)
            factors[name] = compute_discriminative_factor(output, labels)
            sparsity[name] = l0_sparsity(output)
            output = output.int()
            if name in self.cached_output_prev:
                xor = (self.cached_output_prev[name] ^ output).sum(dim=1)
                convergence[name] = xor.float().mean().item() / output.size(1)
            self.cached_output_prev[name] = output
        self.monitor.update_discriminative_factor(factors)
        self.monitor.update_output_convergence(convergence)
        self.monitor.update_sparsity(sparsity)
        self.cached_output.clear()
        self.cached_labels.clear()

    def training_started(self):
        self.monitor.update_weight_sparsity(self.model.weight_sparsity())
        self.monitor.update_weight_dropout(self.model.weight_dropout())

    def _epoch_finished(self, loss):
        kwta_thresholds = self.model.epoch_finished()
        self.monitor.update_kwta_thresholds(kwta_thresholds)
        self.monitor.update_weight_sparsity(self.model.weight_sparsity())
        self.monitor.update_weight_dropout(self.model.weight_dropout())
        self._update_cached()
        TrainerGrad._epoch_finished(self, loss)

    def _on_forward_pass_batch(self, batch, output, train):
        h, y = output
        if train:
            x, labels = batch
            self.cached_labels.append(labels)
            self.cached_output['h'].append(h)
            self.cached_output['y'].append(y)
        TrainerGrad._on_forward_pass_batch(self, batch, y, train)

    def _get_loss(self, batch, output):
        # In case of unsupervised learning, '_get_loss' is overridden
        # accordingly.
        input, labels = batch
        h, y = output
        return self.criterion(y, labels)
