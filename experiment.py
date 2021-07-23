import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from tqdm import trange

from metrics import compute_error, compute_accuracy, cluster_centroids, \
    compute_convergence
from networks import NetworkPermanenceVaryingSparsity

mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.format'] = 'png'
mpl.rcParams['font.size'] = 13
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 14

# The dimensionality of input vector 'x' and output populations 'h' and 'y'
N_x = N_y = N_h = 200

# The initial sparsity of the weights
s_w_xh = s_w_xy = s_w_hy = s_w_yy = s_w_hh = s_w_yh = 0.05

# A color-blind friendly palette
colors = ['#00429d', '#93003a']


def run_experiment(x, labels, network_cls=NetworkPermanenceVaryingSparsity,
                   architecture=('w_xy', 'w_xh', 'w_hy', 'w_hh', 'w_yh'),
                   weights_learn=(),
                   n_iters=20, n_choose=10, lr=0.01,
                   with_accuracy=False, experiment_name=''):
    """
    Run the experiment.

    Parameters
    ----------
    x : (N, S) np.ndarray
        Input samples.
    labels : (S,) np.ndarray
        Sample labels (class ids).
    network_cls : type
        The class of a network to use.
    architecture : list of str
        A list of the connections present in the network.
    weights_learn : list of str
        A list of the connections to learn.
    n_iters : int
        The number of iterations to perform.
    n_choose : int
        The number of non-zero values to choose to update from the pre- and
        post- outer products.
    lr : float
        The learning rate
    with_accuracy : bool
        If True, plot the model accuracy.
    experiment_name : str
        The experiment name.

    Returns
    -------
    network
        The trained network.
    """
    weights = {}
    weights['w_xy'] = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    weights['w_xh'] = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    weights['w_hy'] = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
    weights['w_hh'] = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
    weights['w_yh'] = np.random.binomial(1, s_w_yh, size=(N_h, N_y))
    weights['w_yy'] = np.random.binomial(1, s_w_yy, size=(N_y, N_y))

    for key in weights.keys():
        if key not in architecture:
            # remove these connections
            weights[key] = None

    network = network_cls(weights, weights_learn=weights_learn)

    sparsity = dict(y=np.zeros(n_iters), h=np.zeros(n_iters))
    error = dict(y=np.zeros(n_iters), h=np.zeros(n_iters))
    accuracy = dict(y=np.zeros(n_iters), h=np.zeros(n_iters))
    convergence = dict(y=np.zeros(n_iters), h=np.zeros(n_iters))
    output_prev = dict(y=None, h=None)
    for iter_id in trange(n_iters, desc=network.name):
        h, y = network.train_epoch(x, n_choose=n_choose, lr=lr)
        for name, output in zip(('h', 'y'), (h, y)):
            sparsity[name][iter_id] = output.mean()
            error[name][iter_id] = compute_error(output.T, labels)
            accuracy[name][iter_id] = compute_accuracy(output.T, labels)
            convergence[name][iter_id] = compute_convergence(output, output_prev[name])
            output_prev[name] = output.copy()

    for name in ('h', 'y'):
        print(f"'{name}' final sparsity: {sparsity[name][-1]}")
    for name in weights_learn:
        print(f"{name} final sparsity: {weights[name].mean():.3f}")

    fig, axes = plt.subplots(nrows=2 + with_accuracy, sharex=True)
    axes[-1].set_xlabel("Iteration")
    axes[0].set_ylabel("Error")
    axes[1].set_ylabel("Convergence")
    if with_accuracy:
        axes[2].set_ylabel("Accuracy")
    plt.suptitle(f"{experiment_name.capitalize()}. {network.name}")

    error_x = compute_error(x.T, labels)
    axes[0].axhline(y=error_x, xmin=0, xmax=n_iters - 1, ls='--', color='gray', label="input '$x$'")
    for i, name in enumerate(['h', 'y']):
        axes[0].plot(range(n_iters), error[name], label=f"output '${name}$'", color=colors[i])
        axes[1].plot(range(n_iters), convergence[name], color=colors[i])
        if with_accuracy:
            axes[2].plot(range(n_iters), accuracy[name], color=colors[i])
    axes[0].legend()
    if min(map(np.nanmin, convergence.values())) < 0.05:
        axes[1].set_ylim(ymin=0.)
    axes[-1].set_xlim(xmin=0)
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    results_dir = Path("results") / experiment_name
    results_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(results_dir / f"convergence {network.name}.png", bbox_inches='tight')

    fig, ax = plt.subplots()
    centroids = cluster_centroids(output_prev['y'].T, labels)
    im = ax.imshow(centroids, aspect='auto', interpolation='none', cmap='GnBu')
    plt.colorbar(im)
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Cluster")
    ax.set_title("Mean centroids of 'y'")
    fig.savefig(results_dir / f"centroids {network.name}.png", bbox_inches='tight')
    plt.show()

    return network
