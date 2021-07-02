from collections import defaultdict

import io
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from graph import plot_assemblies
from mighty.monitor.accuracy import calc_accuracy
from mighty.monitor.monitor import MonitorEmbedding, ParamRecord
from mighty.utils.domain import MonitorLevel
from mighty.utils.common import clone_cpu


class ParamRecordBinary(ParamRecord):
    def __init__(self, param, monitor_level=MonitorLevel.DISABLED):
        sign_flips =  torch.zeros(param.shape, dtype=torch.int32)
        if not getattr(param, 'learn', None):
            # not learnable
            monitor_level = MonitorLevel.DISABLED
            sign_flips = None
        super().__init__(param=param, monitor_level=monitor_level)
        if self.prev_data is not None:
            self.prev_data = self.prev_data.int()
        self.sign_flips = sign_flips

    def update_signs(self):
        new_data = clone_cpu(self.param.data.int())
        xor = new_data ^ self.prev_data
        self.sign_flips += xor
        self.prev_data = new_data
        sign_flips_count = xor.sum().item()
        return sign_flips_count


class MonitorIWTA(MonitorEmbedding):
    pos = defaultdict(lambda: None)
    fixed = defaultdict(lambda: None)

    def plot_assemblies(self, assemblies, labels, name=None):
        # take at most 2 classes 2 samples each
        idx = []
        for label in labels.unique()[:2]:
            take = (labels == label).nonzero(as_tuple=True)[0][:2]
            idx.extend(take.tolist())
        assemblies = assemblies[idx]
        labels = labels[idx]
        nonzero = [vec.nonzero(as_tuple=True)[0].cpu().numpy()
                   for vec in assemblies.cpu()]
        ax, self.pos[name], self.fixed[name] = plot_assemblies(
            nonzero,
            pos=self.pos[name],
            fixed=self.fixed[name],
            labels=labels.tolist()
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

    def update_weight_sparsity(self, sparsity: dict):
        names, sparsity = zip(*sparsity.items())
        self.viz.line_update(y=sparsity, opts=dict(
            xlabel='Epoch',
            ylabel='L1 norm / size',
            title="Weight L1 sparsity",
            legend=list(names),
        ))

    def update_weight_dropout(self, dropout: dict):
        names, dropout = zip(*dropout.items())
        self.viz.line_update(y=dropout, opts=dict(
            xlabel='Epoch',
            title="Weight dropout",
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

    def update_kwta_thresholds(self, kwta_thresholds: dict):
        labels, thresholds = zip(*kwta_thresholds.items())
        self.viz.line_update(thresholds, opts=dict(
            xlabel="Epoch",
            ylabel="Threshold",
            title="kWTA permanence thresholds",
            legend=list(labels),
        ))

    def epoch_finished(self):
        super().epoch_finished()
        self.update_sign_flips_hist()

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

    def update_discriminative_factor(self, factors: dict):
        labels, factors = zip(*factors.items())
        self.viz.line_update(y=factors, opts=dict(
            xlabel='Epoch',
            ylabel='dist-other / dist-same',
            title="Clustering discriminative factor",
            legend=list(labels),
        ))

    def update_output_convergence(self, convergence: dict):
        if len(convergence) == 0:
            return
        labels, convergence = zip(*convergence.items())
        self.viz.line_update(y=convergence, opts=dict(
            xlabel='Epoch',
            ylabel="mean(y ^ y_prev)",
            title="Output convergence",
            legend=list(labels),
        ))

    def update_sparsity_per_label(self, sparsity: dict):
        if sparsity["p(x)"] is None:
            return
        labels, sparsity = zip(*sparsity.items())
        sparsity = np.stack(sparsity, axis=1)
        title = "Habituation"
        self.viz.bar(sparsity, win=title, opts=dict(
            stacked=False,
            legend=list(labels),
            title=title
        ))
