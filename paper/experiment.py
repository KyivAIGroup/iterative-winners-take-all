import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import trange

from metrics import compute_loss, compute_accuracy, cluster_centroids, compute_convergence
from networks import NetworkPermanenceVaryingSparsity

N_x = N_y = N_h = 200
s_x = 0.2
s_w_xh = s_w_xy = s_w_hy = s_w_yy = s_w_hh = s_w_yh = 0.05


def run_experiment(x, labels, network_cls=NetworkPermanenceVaryingSparsity,
                   n_iters=20, n_choose=10, lr=0.01,
                   with_accuracy=False, experiment_name=''):
    weights = {}
    weights['w_xy'] = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    weights['w_xh'] = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    weights['w_hy'] = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
    weights['w_hh'] = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
    weights['w_yh'] = np.random.binomial(1, s_w_yh, size=(N_h, N_y))
    weights['w_yy'] = None

    network = network_cls(weights)

    sparsity = dict(y=np.zeros(n_iters), h=np.zeros(n_iters))
    loss = dict(y=np.zeros(n_iters), h=np.zeros(n_iters))
    accuracy = dict(y=np.zeros(n_iters), h=np.zeros(n_iters))
    convergence = dict(y=np.zeros(n_iters), h=np.zeros(n_iters))
    output_prev = dict(y=None, h=None)
    for iter_id in trange(n_iters, desc=network.name):
        h, y = network.train_epoch(x, n_choose=n_choose, lr=lr)
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

    fig, axes = plt.subplots(nrows=2 + with_accuracy, sharex=True)
    axes[-1].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Convergence")
    if with_accuracy:
        axes[2].set_ylabel("Accuracy")
    # plt.suptitle(f"{network.name} {experiment_name}")

    loss_x = compute_loss(x.T, labels)
    axes[0].axhline(y=loss_x, xmin=0, xmax=n_iters - 1, ls='--', color='gray', label="input 'x'")
    for name in ('h', 'y'):
        line = axes[0].plot(range(n_iters), loss[name], label=f"output '{name}'")[0]
        axes[1].plot(range(n_iters), convergence[name], color=line.get_color())
        if with_accuracy:
            axes[2].plot(range(n_iters), accuracy[name], color=line.get_color())
    axes[0].legend()
    plt.tight_layout()

    results_dir = Path("results") / experiment_name
    results_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(results_dir / f"convergence {network.name}.png", dpi=300)

    fig, ax = plt.subplots()
    centroids = cluster_centroids(output_prev['y'].T, labels)
    im = ax.imshow(centroids, aspect='auto', interpolation='none')
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Class label")
    # ax.set_title("Mean centroids of 'y'")
    plt.colorbar(im)
    fig.savefig(results_dir / f"centroids {network.name}.png", dpi=300)
    # plt.show()
