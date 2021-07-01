from collections import defaultdict

import io
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import plot_assemblies
from mighty.monitor.accuracy import calc_accuracy
from mighty.monitor.monitor import MonitorEmbedding, ParamRecord
from mighty.utils.domain import MonitorLevel


class ParamRecordBinary(ParamRecord):
    def __init__(self, param, monitor_level=MonitorLevel.DISABLED):
        sign_flips =  torch.zeros(param.shape, dtype=torch.int32)
        if not getattr(param, 'learn', None):
            # not learnable
            monitor_level = MonitorLevel.DISABLED
            sign_flips = None
        super().__init__(param=param, monitor_level=monitor_level)
        self.sign_flips = sign_flips

    @staticmethod
    def count_sign_flips(new_data, prev_data):
        return (new_data ^ prev_data).sum().item()

    def update_signs(self):
        if self.prev_data is not None:
            self.sign_flips += self.param.data.cpu() ^ self.prev_data
        sign_flips_count = super().update_signs()
        return sign_flips_count


class MonitorIWTA(MonitorEmbedding):
    pos = defaultdict(lambda: None)
    fixed = defaultdict(lambda: None)
    iteration = 0
    track_iwta = False
    iwta_activations = []

    def plot_assemblies(self, assemblies, labels=None, name=None):
        if labels is not None:
            # take at most 2 classes 2 samples each
            idx = []
            for class_id in labels.unique()[:2]:
                take = (labels == class_id).nonzero(as_tuple=True)[0][:2]
                idx.extend(take.tolist())
            assemblies = assemblies[idx]
            labels = labels[idx]
        nonzero = [vec.nonzero(as_tuple=True)[0].numpy()
                   for vec in assemblies.cpu()]
        ax, self.pos[name], self.fixed[name] = plot_assemblies(
            nonzero,
            pos=self.pos[name],
            fixed=self.fixed[name],
            labels=labels
        )
        with io.BytesIO() as buff:
            ax.figure.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = ax.figure.canvas.get_width_height()
        plt.close(ax.figure)
        image = data.reshape((int(h), int(w), -1)).transpose((2, 0, 1))
        self.viz.image(image, win=f'assembly-{name}', opts=dict(
            caption=f"{name} [{self.timer.epoch}]",
            store_history=True
        ))

    def reset(self):
        # don't reset precords
        pass

    def iwta_iteration(self, z_h, z_y, id_=0):
        if not self.track_iwta:
            return
        # self._plot_iwta_scatter(z_h, z_y, id_=id_)
        z_h = z_h[id_].detach().cpu().type(torch.float32).clone()
        z_y = z_y[id_].detach().cpu().type(torch.float32).clone()
        if len(self.iwta_activations) > 0:
            z_h_prev, z_y_prev = self.iwta_activations[-1]
            z_h[z_h_prev.nonzero(as_tuple=True)] = 0.5
            z_y[z_y_prev.nonzero(as_tuple=True)] = 0.5
            z_h_prev[((z_h_prev == 1) & (z_h == 0)).nonzero(as_tuple=True)] = np.nan
            z_y_prev[((z_y_prev == 1) & (z_y == 0)).nonzero(as_tuple=True)] = np.nan
        self.iwta_activations.append([z_h, z_y])

    def update_weight_sparsity(self, sparsity: dict):
        names, sparsity = list(zip(*sparsity.items()))
        self.viz.line_update(y=sparsity, opts=dict(
            xlabel='Epoch',
            ylabel='L1 norm / size',
            title="Weight L1 sparsity",
            legend=list(names),
        ))

    def update_sign_flips_hist(self):
        for name, precord in self.param_records.items():
            if precord.sign_flips is None:
                continue
            sign_flips = precord.sign_flips.flatten()
            self.viz.histogram(sign_flips, win=f"{name}-sflip", opts=dict(
                xlabel="No. of sign flips",
                ylabel="Count",
                title=f"{name} instability",
                ytype='log',
            ))

    def epoch_finished(self):
        super().epoch_finished()
        self.update_sign_flips_hist()
        if len(self.iwta_activations) == 0:
            return
        z_h, z_y = self.iwta_activations[-1]
        if z_y.any() or z_h.any():
            # don't add zero space many times
            self.iwta_activations.append([torch.zeros_like(z_h),
                                          torch.zeros_like(z_y)])
        self.iteration = 0

    def _plot_iwta_scatter(self, z_h, z_y, id_):
        for name, assembly in dict(z_h=z_h, z_y=z_y).items():
            assembly = assembly[id_]
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
        if len(self.iwta_activations) == 0:
            return
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
            self.param_records[name] = ParamRecordBinary(
                param,
                monitor_level=self._advanced_monitoring_level
            )

    def embedding_hist(self, activations):
        pass

    def update_accuracy_epoch(self, labels_pred, labels_true, mode):
        accuracy = calc_accuracy(labels_true, labels_pred)
        self.update_accuracy(accuracy=accuracy, mode=mode)
        return accuracy

    def update_weight_histogram(self):
        if not self._advanced_monitoring_level & MonitorLevel.WEIGHT_HISTOGRAM:
            return
        for name, precord in self.param_records.items():
            permanence = getattr(precord.param, "permanence", None)
            learn = getattr(precord.param, "learn", None)
            if permanence is None or not learn:
                continue
            self.viz.histogram(X=permanence.view(-1), win=name, opts=dict(
                xlabel='Permanence',
                ylabel='Count',
                title=name,
            ))

    def update_pairwise_similarity(self, tensor, labels, name=''):
        tensor = tensor.float()
        for label in labels.unique().tolist():
            t = tensor[labels == label]
            n_elem = len(t)
            if n_elem == 1:
                continue
            cos = F.cosine_similarity(t.unsqueeze(1), t.unsqueeze(0), dim=2)
            ii, jj = torch.triu_indices(row=n_elem, col=n_elem, offset=1)
            cos = cos[ii, jj]
            win = f"{name} label={label}"
            self.viz.histogram(X=cos, win=win, opts=dict(
                xlabel='Cosine similarity',
                ylabel='Count',
                numbins=20,
                xtickmin=0,
                xtickmax=1,
                title=win,
            ))

    def update_discriminative_factor(self, factor: float):
        if factor is not None:
            self.viz.line_update(y=factor, opts=dict(
                xlabel='Epoch',
                ylabel='dist-other / dist-same',
                title="Clustering discriminative factor",
            ))
