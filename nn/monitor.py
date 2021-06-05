import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import io
import numpy as np
import torch.nn as nn
from collections import defaultdict

from graph import plot_assemblies
from mighty.monitor.accuracy import calc_accuracy
from mighty.monitor.monitor import MonitorEmbedding, ParamRecord


class MonitorIWTA(MonitorEmbedding):
    pos = defaultdict(lambda: None)
    fixed = defaultdict(lambda: None)

    def plot_assemblies(self, assemblies, name=None):
        ax, self.pos[name], self.fixed[name] = plot_assemblies(
            assemblies.cpu().T.numpy(),
            pos=self.pos[name],
            fixed=self.fixed[name])
        with io.BytesIO() as buff:
            ax.figure.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = ax.figure.bbox.bounds[2:]
        plt.close(ax.figure)
        image = data.reshape((int(h), int(w), -1)).transpose((2, 0, 1))
        self.viz.image(image, win=f'assembly-{name}', opts=dict(
            caption=f"[name='{name}'] Epoch {self.timer.epoch}",
            store_history=True
        ))

    def register_layer(self, layer: nn.Module, prefix: str):
        """
        Register a layer to monitor.

        Parameters
        ----------
        layer : nn.Module
            A model layer.
        prefix : str
            The layer name.
        """
        for name, param in layer.named_parameters(prefix=prefix):
            self.param_records[name] = ParamRecord(
                param,
                monitor_level=self._advanced_monitoring_level
            )

    def update_accuracy_epoch(self, labels_pred, labels_true, mode):
        """
        The callback to calculate and update the epoch accuracy from a batch
        of predicted and true class labels.

        Parameters
        ----------
        labels_pred, labels_true : (N,) torch.Tensor
            Predicted and true class labels.
        mode : str
            Update mode: 'batch' or 'epoch'.

        Returns
        -------
        accuracy : torch.Tensor
            A scalar tensor with one value - accuracy.
        """
        accuracy = calc_accuracy(labels_true, labels_pred)
        self.update_accuracy(accuracy=accuracy, mode=mode)
        return accuracy
