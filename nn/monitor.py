import torch

from mighty.monitor.monitor import MonitorEmbedding
from graph import plot_assemblies
import matplotlib.pyplot as plt


class MonitorIWTA(MonitorEmbedding):

    def plot_assemblies(self, assemblies):
        ax = plot_assemblies(assemblies.cpu().T.numpy(),
                             title=f"iWTA. Epoch {self.timer.epoch}")
        self.viz.matplot(plt)
