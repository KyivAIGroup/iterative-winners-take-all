"""
For a valid comparison with kWTA, only w_hy is learned.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import trange

from kwta import iWTA, iWTA_history, kWTA
from permanence import PermanenceVogels, PermanenceFixedSparsity, \
    PermanenceVaryingSparsity, ParameterBinary
from metrics import compute_loss, compute_accuracy, cluster_centroids, compute_convergence

# Fix the random seed to reproduce the results
np.random.seed(0)

N_x = N_y = N_h = 200
s_x = 0.2
s_w_xh = s_w_xy = s_w_hy = s_w_yy = s_w_hh = s_w_yh = 0.05

N_ITERS = 50
N_CHOOSE = 10
LEARNING_RATE = 0.01

N_CLASSES = 10
N_SAMPLES_PER_CLASS = 100


class ExperimentWillshaw:
    name = "Classical Willshaw iWTA"

    def __init__(self, weights: dict):
        self.weights = {}
        for name, w in weights.items():
            self.weights[name] = ParameterBinary(w)

    def train_epoch(self, x):
        h, y = iWTA(x, **self.weights)
        self.weights['w_xy'].update(x_pre=x, x_post=y, n_choose=N_CHOOSE)
        self.weights['w_xh'].update(x_pre=x, x_post=h, n_choose=N_CHOOSE)
        self.weights['w_hh'].update(x_pre=h, x_post=h, n_choose=N_CHOOSE)
        self.weights['w_hy'].update(x_pre=h, x_post=y, n_choose=N_CHOOSE)
        if self.weights['w_yy'] is not None:
            self.weights['w_yy'].update(x_pre=y, x_post=y, n_choose=N_CHOOSE)
        self.weights['w_yh'].update(x_pre=y, x_post=h, n_choose=N_CHOOSE)
        return h, y


class ExperimentKWTA:
    name = "Classical Willshaw kWTA"
    K_FIXED = 10

    def __init__(self, weights: dict):
        self.weights = {}
        for name in ("w_xy", "w_xh", "w_hy"):
            self.weights[name] = ParameterBinary(weights[name])

    def train_epoch(self, x):
        h = kWTA(self.weights['w_xh'] @ x, k=self.K_FIXED)
        y = kWTA(self.weights['w_xy'] @ x - self.weights['w_hy'] @ h, k=self.K_FIXED)
        self.weights['w_xy'].update(x_pre=x, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        self.weights['w_xh'].update(x_pre=x, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        self.weights['w_hy'].update(x_pre=h, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        return h, y


class ExperimentPermanenceFixedSparsity:
    name = "Permanence fixed sparsity"

    def __init__(self, weights: dict):
        self.weights = {}
        for name, w in weights.items():
            self.weights[name] = PermanenceFixedSparsity(w)

    def train_epoch(self, x):
        h, y = iWTA(x, **self.weights)
        self.weights['w_xy'].update(x_pre=x, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        self.weights['w_xh'].update(x_pre=x, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        self.weights['w_hh'].update(x_pre=h, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        self.weights['w_hy'].update(x_pre=h, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        if self.weights['w_yy'] is not None:
            self.weights['w_yy'].update(x_pre=y, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        self.weights['w_yh'].update(x_pre=y, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        return h, y


class ExperimentPermanenceVaryingSparsity(ExperimentPermanenceFixedSparsity):
    name = "Permanence varying sparsity"

    def __init__(self, weights: dict, output_sparsity_desired=(0.025, 0.1)):
        self.weights = {}
        self.weights['w_xh'] = PermanenceVaryingSparsity(weights['w_xh'],
                                                         excitatory=True,
                                                         output_sparsity_desired=output_sparsity_desired)
        self.weights['w_xy'] = PermanenceVaryingSparsity(weights['w_xy'],
                                                         excitatory=True,
                                                         output_sparsity_desired=output_sparsity_desired)
        self.weights['w_hh'] = PermanenceVaryingSparsity(weights['w_hh'],
                                                         excitatory=False,
                                                         output_sparsity_desired=output_sparsity_desired)
        self.weights['w_hy'] = PermanenceVaryingSparsity(weights['w_hh'],
                                                         excitatory=False,
                                                         output_sparsity_desired=output_sparsity_desired)
        self.weights['w_yy'] = PermanenceVaryingSparsity(weights['w_yy'],
                                                         excitatory=True,
                                                         output_sparsity_desired=output_sparsity_desired)
        self.weights['w_yh'] = PermanenceVaryingSparsity(weights['w_yh'],
                                                         excitatory=True,
                                                         output_sparsity_desired=output_sparsity_desired)


class ExperimentPermanenceVogels:
    name = "Permanence Vogels"

    def __init__(self, weights: dict):
        self.weights = {}
        for name in ("w_xy", "w_xh", "w_yy", "w_yh"):
            self.weights[name] = PermanenceFixedSparsity(weights[name])
        for name in ("w_hy", "w_hh"):
            self.weights[name] = PermanenceVogels(weights[name])

    def train_epoch(self, x):
        z_h, z_y = iWTA_history(x, **self.weights)
        h, y = z_h[0], z_y[0]
        for i in range(1, len(z_h)):
            h |= z_h[i]
            y |= z_y[i]

        self.weights['w_xy'].update(x_pre=x, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        self.weights['w_xh'].update(x_pre=x, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        self.weights['w_hh'].update(x_pre=z_h, x_post=z_h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        self.weights['w_hy'].update(x_pre=z_h, x_post=z_y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        if self.weights['w_yy'] is not None:
            self.weights['w_yy'].update(x_pre=y, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        self.weights['w_yh'].update(x_pre=y, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)

        return h, y


x_centroids = np.random.binomial(1, s_x, size=(N_CLASSES, N_x))
labels = np.repeat(np.arange(N_CLASSES), N_SAMPLES_PER_CLASS)
np.random.shuffle(labels)
x = x_centroids[labels].T
white_noise = np.random.binomial(1, 0.5 * s_x, size=x.shape)
x ^= white_noise
print(f"'x' accuracy: {compute_accuracy(x.T, labels)}")


for experiment_cls in (ExperimentPermanenceVaryingSparsity, ExperimentWillshaw, ExperimentKWTA, ExperimentPermanenceFixedSparsity, ExperimentPermanenceVogels):
    weights = {}
    weights['w_xy'] = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    weights['w_xh'] = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    weights['w_hy'] = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
    weights['w_hh'] = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
    weights['w_yh'] = np.random.binomial(1, s_w_yh, size=(N_h, N_y))
    weights['w_yy'] = None

    experiment = experiment_cls(weights)

    sparsity = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    loss = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    accuracy = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    convergence = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    output_prev = dict(y=None, h=None)
    for iter_id in trange(N_ITERS, desc=experiment.name):
        h, y = experiment.train_epoch(x)
        for name, output in zip(('h', 'y'), (h, y)):
            sparsity[name][iter_id] = output.mean()
            loss[name][iter_id] = compute_loss(output.T, labels)
            accuracy[name][iter_id] = compute_accuracy(output.T, labels)
            convergence[name][iter_id] = compute_convergence(output, output_prev[name])
            output_prev[name] = output.copy()

    for name in ('h', 'y'):
        print(f"'{name}' final sparsity: {sparsity[name][-1]}")
    for name, w in weights.items():
        if w is not None:
            print(f"{name} final sparsity: {w.mean():.3f}")

    fig, axes = plt.subplots(nrows=3, sharex=True)
    axes[-1].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")
    axes[2].set_ylabel("Convergence")
    plt.suptitle(f"{experiment.name} ($s_y$={sparsity['y'][-1]:.3f})")

    loss_x = compute_loss(x.T, labels)
    axes[0].axhline(y=loss_x, xmin=0, xmax=N_ITERS - 1, ls='--', color='gray', label="Input 'x'")
    for name in ('h', 'y'):
        line = axes[0].plot(range(N_ITERS), loss[name], label=f"Output '{name}'")[0]
        axes[1].plot(range(N_ITERS), accuracy[name], color=line.get_color())
        axes[2].plot(range(N_ITERS), convergence[name], color=line.get_color())
    axes[0].legend()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    fig.savefig(results_dir / f"clustering {experiment.name}.png", dpi=300)

    fig, ax = plt.subplots()
    centroids = cluster_centroids(output_prev['y'].T, labels)
    im = ax.imshow(centroids, aspect='auto', interpolation='none')
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Class label")
    ax.set_title("Mean centroids of 'y'")
    plt.colorbar(im)
    fig.savefig(results_dir / f"centroids {experiment.name}.png", dpi=300)
    plt.show()
