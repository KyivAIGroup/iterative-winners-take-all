"""
Habituation experiment shows that the output sparsity encodes information
about the input data distribution (frequency of encountering).
"""
import matplotlib as mpl

import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from tqdm import trange

from kwta import iWTA, iWTA_history
from metrics import compute_convergence
from permanence import *

mpl.rcParams['savefig.dpi'] = 800
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['font.size'] = 13
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 14

# Fix the random seed to reproduce the results
np.random.seed(0)
# plt.style.use('ggplot')

N_x = N_h = N_y = 200
s_x = 0.2
s_w_xy = s_w_xh = s_w_hy = s_w_hh = 0.05
s_w_yy = s_w_yh = 0.01
N_REPEATS = 5
N_CHOOSE = 10
LEARNING_RATE = 0.01
N_SAMPLES_TOTAL = 10  # 2 samples of x_0 and x_1 and 6 of x_2
px = [0.2, 0.2, 0.6]  # probability of encountering x_0, x_1, and x_2


def generate_k_active(n, k):
    """
    Sample a random binary vector of size `n` with exactly `k` ones.

    Parameters
    ----------
    n : int
        The size of a vector.
    k : int
        The number of non-zero values.

    Returns
    -------
    x : (n,) np.ndarray
        A binary vector.
    """
    x = np.zeros(n, dtype=np.int32)
    active = np.random.choice(n, size=k, replace=False)
    x[active] = 1
    return x


def sample_from_distribution(px, n_neurons, n_samples, k):
    """
    Sample `n_samples` vectors of size `n_neurons` from the `px` distribution.
    Each binary vector has exactly `k` ones.

    Parameters
    ----------
    px : list
        The probability of encountering specific `x_i` stimulus from a set
        of input stimuli.
    n_neurons : int
        The vector size.
    n_samples : int
        The number of samples to generate.
    k : int
        The number of non-zero entries in a vector.

    Returns
    -------
    x : (n_neurons, n_samples) np.ndarray
        Data samples with duplicate vectors.
    """
    px = np.array(px)
    assert np.isclose(px.sum(), 1), "Probabilities must sum up to 1"
    x = np.array([generate_k_active(n_neurons, k) for pxi in px])
    labels = []
    for i, pxi in enumerate(px):
        repeats = math.ceil(pxi * n_samples)
        labels_repeated = np.full(repeats, fill_value=i)
        labels.append(labels_repeated)
    labels = np.hstack(labels)
    # Shuffling is not required; we do this to illustrate that the obtained
    # results are not due to the sequential nature of the input data.
    np.random.shuffle(labels)
    x = x[labels].T
    return x, labels


for perm_cls in (PermanenceVaryingSparsity, ParameterBinary, PermanenceFixedSparsity, PermanenceVogels):
    N_ITERS = 6 if perm_cls is ParameterBinary else 15
    y_sparsity = np.zeros((N_REPEATS, N_ITERS, len(px)), dtype=np.float32)
    convergence = np.zeros((N_REPEATS, N_ITERS))
    weight_sparsity = {"w_hy": np.zeros(N_REPEATS)}

    y_unique = None
    for repeat in trange(N_REPEATS, desc=perm_cls.__name__):
        x, labels = sample_from_distribution(px=px, n_neurons=N_x,
                                             n_samples=N_SAMPLES_TOTAL,
                                             k=int(s_x * N_x))

        w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
        w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
        w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
        w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
        w_yy = np.random.binomial(1, s_w_yy, size=(N_y, N_y))
        w_yh = np.random.binomial(1, s_w_yh, size=(N_h, N_y))

        # Train w_hy only
        w_hy = perm_cls(w_hy, excitatory=False)

        y_prev = None
        for iter_id in range(N_ITERS):
            if perm_cls is PermanenceVogels:
                z_h, z_y = iWTA_history(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy, w_yh=w_yh)
                w_hy.update(x_pre=z_h, x_post=z_y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
                h, y = z_h[0], z_y[0]
                for i in range(1, len(z_h)):
                    h |= z_h[i]
                    y |= z_y[i]
            else:
                h, y = iWTA(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy, w_yh=w_yh)
                w_hy.update(x_pre=h, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
            y_sparsity_i = y.mean(axis=0)
            for label in range(len(px)):
                mask = labels == label
                y_sparsity[repeat, iter_id, label] = y_sparsity_i[mask].mean()
            convergence[repeat, iter_id] = compute_convergence(y, y_prev)
            y_prev = y.copy()

        if repeat == 0:
            _, idx_unique = np.unique(labels, return_index=True)
            y_unique = y_prev.T[idx_unique]

        weight_sparsity["w_hy"][repeat] = w_hy.mean()

    for w_name, w_sparsity in weight_sparsity.items():
        print(f"{w_name} final sparsity: {w_sparsity.mean():.3f}")

    results_dir = Path("results") / "habituation"
    results_dir.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_aspect(15)
    ax.eventplot([y.nonzero()[0] for y in y_unique], colors='black', linelengths=0.8)
    ax.set_yticks(range(len(px)))
    ax.set_yticklabels([f"$y(x_{i})$" for i in range(len(px))])
    ax.set_xticks([0, N_y - 1])
    ax.set_xticklabels(['1', str(N_y)])
    ax.set_title("Habituation rasterplot")
    ax.set_xlabel("Neuron")
    ax.xaxis.set_label_coords(0.5, -0.03)
    fig.savefig(results_dir / f"rasterplot {perm_cls.__name__}.pdf", bbox_inches='tight')

    fig, ax = plt.subplots(nrows=1 + (perm_cls is not ParameterBinary), sharex=True)
    ax = np.atleast_1d(ax)
    mean = y_sparsity.mean(axis=0)
    std = y_sparsity.std(axis=0)
    print(f"'y' final sparsity: {mean[-1]}")
    for label, (m, s) in enumerate(zip(mean.T, std.T)):
        ax[0].plot(range(N_ITERS), m, label=f"$x_{label}$")
        ax[0].fill_between(range(N_ITERS), m + s, m - s, alpha=0.2)
    ax[0].legend()
    ax[-1].set_xlabel("Iteration")
    ax[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_ylabel("$s_y$")
    ax[-1].set_xlim(xmin=0)
    ax[0].set_ylim(ymin=0)

    if len(ax) > 1:
        ax[1].set_ylabel("Convergence")
        mean = convergence.mean(axis=0)
        std = convergence.std(axis=0)
        ax[1].plot(range(N_ITERS), mean, label='$y$')
        ax[1].fill_between(range(N_ITERS), mean + std, mean - std, alpha=0.2)
        ax[1].set_ylim(ymin=0)
        ax[1].legend()
    plt.tight_layout()
    fig.savefig(results_dir / f"{perm_cls.__name__}.pdf", bbox_inches='tight')

    plt.show()
