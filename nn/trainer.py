import torch.nn as nn

from mighty.trainer import TrainerEmbedding
from mighty.utils.data import DataLoader
from mighty.utils.stub import OptimizerStub
from mighty.monitor.accuracy import AccuracyEmbedding

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
        return y

    def train_batch(self, batch):
        h, y = self.model(batch[0])
        self.model.update_weights(h, y)
        loss = self._get_loss(batch, y)
        return loss

    def _epoch_finished(self, loss):
        x, labels = self.data_loader.sample()
        h, y = self.model(x)
        self.monitor.plot_assemblies(y)
        super()._epoch_finished(loss)
