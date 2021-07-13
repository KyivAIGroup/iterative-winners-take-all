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
from nn_utils import PERMANENCE_INSTABILITY


class ParamRecordBinary(ParamRecord):
    def __init__(self, param, monitor_level=MonitorLevel.DISABLED):
        sign_flips =  torch.zeros(param.shape, dtype=torch.int32)
        if not getattr(param, 'learn', None):
            # not learnable
            monitor_level = MonitorLevel.DISABLED
            sign_flips = None
        if not (monitor_level & PERMANENCE_INSTABILITY):
            sign_flips = None
        super().__init__(param=param, monitor_level=monitor_level)
        if self.prev_data is not None:
            self.prev_data = self.prev_data.int()
        self.sign_flips = sign_flips

    def update_signs(self):
        new_data = clone_cpu(self.param.data.int())
        xor = new_data ^ self.prev_data
        if self.sign_flips is not None:
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

    def update_weight_nonzero_keep(self, nonzero_keep: dict):
        if len(nonzero_keep) == 0:
            return
        names, nonzero_keep = zip(*nonzero_keep.items())
        self.viz.line_update(y=nonzero_keep, opts=dict(
            xlabel="Epoch",
            ylabel="1.0 - dropout",
            title="Weight nonzero keep",
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
        if len(kwta_thresholds) == 0:
            return
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

    def weights_heatmap(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.learn:
                continue
            self.viz.heatmap(param.data.flipud(), win=f"{name} heatmap", opts=dict(
                title=f"{name} binary",
                xlabel='Neuron Output',
                ylabel='Neuron Input',
                ytick=False,
            ))
            continue
            self.viz.heatmap(param.permanence.flipud(), win=f"{name} perm", opts=dict(
                title=f"{name} permanence",
                xlabel='Neuron Output',
                ylabel='Neuron Input',
                ytick=False,
            ))

    def update_weight_histogram(self):
        if not self._advanced_monitoring_level & MonitorLevel.WEIGHT_HISTOGRAM:
            return
        for name, precord in self.param_records.items():
            permanence = getattr(precord.param, "permanence", None)
            learn = getattr(precord.param, "learn", None)
            if permanence is None or not learn:
                continue
            permanence = permanence.view(-1)
            permanence = permanence[permanence.nonzero(as_tuple=True)[0]]
            self.viz.histogram(X=permanence, win=name, opts=dict(
                xlabel='Permanence',
                ylabel='Count',
                title=name,
                ytype='log',
                xtype='log',
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

    def update_sparsity(self, sparsity, mode=None):
        if isinstance(sparsity, dict):
            labels, sparsity = zip(*sparsity.items())
            self.viz.line_update(y=sparsity, opts=dict(
                xlabel='Epoch',
                ylabel='||y||_0 / size(y)',
                title="Output L0 sparsity",
                legend=list(labels),
            ))
        else:
            super().update_sparsity(sparsity, mode)

    def update_loss(self, loss, mode='batch'):
        if mode.startswith("pairwise"):
            super().update_loss(loss, mode=mode)

    def update_contribution(self, contribution: dict):
        labels, contribution = zip(*contribution.items())
        contribution = torch.stack(contribution)
        self.viz.heatmap(contribution, win="contribution", opts=dict(
            title="Weight contribution",
            rownames=list(labels),
            xlabel='Neuron Output',
        ))
