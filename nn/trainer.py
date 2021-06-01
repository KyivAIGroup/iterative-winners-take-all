import torch.nn as nn

from mighty.trainer import TrainerEmbedding
from mighty.utils.data import DataLoader
from mighty.utils.stub import OptimizerStub
from mighty.monitor.accuracy import AccuracyEmbedding

from nn.monitor import MonitorIWTA


class TrainerIWTA(TrainerEmbedding):

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 env_suffix='',
                 **kwargs):
        TrainerEmbedding.__init__(self, model=model,
                                  criterion=criterion,
                                  data_loader=data_loader,
                                  optimizer=OptimizerStub(),
                                  scheduler=None,
                                  accuracy_measure=AccuracyEmbedding(),
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

    def _get_loss(self, batch, output):
        h, y = output
        return super()._get_loss(batch, output=y)

    def train_batch(self, batch):
        output = self._forward(batch)
        loss = self._get_loss(batch, output)
        return loss

    def _on_forward_pass_batch(self, batch, output, train):
        h, y = output
        super()._on_forward_pass_batch(batch, output=y, train=train)

    def _epoch_finished(self, loss):
        x = self.data_loader.sample()[0]
        h, y = self._forward(x)
        self.monitor.plot_assemblies(y)
        super()._epoch_finished(loss)
