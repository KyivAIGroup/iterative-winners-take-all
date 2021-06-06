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
    iteration = 0
    track_iwta = False
    iwta_activations = []

    def plot_assemblies(self, assemblies, name=None):
        nonzero = [vec.nonzero(as_tuple=True)[0].numpy()
                   for vec in assemblies.cpu()]
        ax, self.pos[name], self.fixed[name] = plot_assemblies(
            nonzero,
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
            caption=f"{name} [{self.timer.epoch}]",
            store_history=True
        ))

    def iwta_iteration(self, z_h, z_y, id_=0):
        if not self.track_iwta:
            return
        # self._plot_iwta_scatter(z_h, z_y)
        z_h = z_h[id_].cpu().type(torch.float32)
        z_y = z_y[id_].cpu().type(torch.float32)
        if len(self.iwta_activations) > 0:
            z_h_prev, z_y_prev = self.iwta_activations[-1]
            z_h[z_h_prev.nonzero(as_tuple=True)] = 0.5
            z_y[z_y_prev.nonzero(as_tuple=True)] = 0.5
        self.iwta_activations.append([z_h, z_y])

    def update_weight_sparsity(self, sparsity: dict):
        names, sparsity = list(zip(*sparsity.items()))
        self.viz.line_update(y=sparsity, opts=dict(
            xlabel='Epoch',
            ylabel='L1 norm / size',
            title="Weight L1 sparsity",
            legend=list(names),
        ))

    def epoch_finished(self):
        super().epoch_finished()
        z_h, z_y = self.iwta_activations[-1]
        self.iwta_activations.append([torch.zeros_like(z_h),
                                      torch.zeros_like(z_y)])
        self.iteration = 0

    def batch_finished(self, model):
        super().batch_finished(model)

    def _plot_iwta_scatter(self, z_h, z_y):
        for name, assembly in dict(z_h=z_h, z_y=z_y).items():
            size = assembly.size(0)
            assembly = assembly.nonzero(as_tuple=True)[0]
            xs = np.full(len(assembly), self.iteration, dtype=np.float32)
            coords = np.c_[xs, assembly]
            win = f"{name} [{self.timer.epoch}]"
            self.viz.scatter(coords, win=win, update='append', opts=dict(
                markersize=3,
                ylabel='Neuron',
                xlabel='Iteration',
                title=win,
                ytickmin=0,
                ytickmax=size,
            ))
            self.viz.update_window_opts(win=win,
                                        opts=dict(
                                            xtickmin=-0.2,
                                            xtickmax=self.iteration + 0.2
                                        ))
        self.iteration += 1

    def _plot_iwta_heatmap(self):
        z_h, z_y = list(zip(*self.iwta_activations))
        z_h = np.stack(z_h, axis=1).astype(np.float32)
        z_y = np.stack(z_y, axis=1).astype(np.float32)
        for name, assembly in dict(z_h=z_h, z_y=z_y).items():
            self.viz.heatmap(assembly, win=f"{name}-heatmap", opts=dict(
                title=name,
                xlabel='Iteration',
                ylabel='Neuron',
            ))
        self.iwta_activations.clear()

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

    def embedding_hist(self, activations):
        pass

    def update_accuracy_epoch(self, labels_pred, labels_true, mode):
        accuracy = calc_accuracy(labels_true, labels_pred)
        # self.update_accuracy(accuracy=accuracy, mode=mode)
        return accuracy
