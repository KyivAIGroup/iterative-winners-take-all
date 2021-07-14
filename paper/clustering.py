import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import trange

from networks import *
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


x_centroids = np.random.binomial(1, s_x, size=(N_CLASSES, N_x))
labels = np.repeat(np.arange(N_CLASSES), N_SAMPLES_PER_CLASS)
np.random.shuffle(labels)
x = x_centroids[labels].T
white_noise = np.random.binomial(1, 0.5 * s_x, size=x.shape)
x ^= white_noise
print(f"'x' accuracy: {compute_accuracy(x.T, labels)}")

for network_cls in (NetworkPermanenceVaryingSparsity,
                    NetworkWillshaw,
                    NetworkKWTA,
                    NetworkPermanenceFixedSparsity,
                    NetworkPermanenceVogels):
    weights = {}
    weights['w_xy'] = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    weights['w_xh'] = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    weights['w_hy'] = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
    weights['w_hh'] = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
    weights['w_yh'] = np.random.binomial(1, s_w_yh, size=(N_h, N_y))
    weights['w_yy'] = None

    network = network_cls(weights)

    sparsity = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    loss = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    accuracy = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    convergence = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    output_prev = dict(y=None, h=None)
    for iter_id in trange(N_ITERS, desc=network.name):
        h, y = network.train_epoch(x, n_choose=N_CHOOSE, lr=LEARNING_RATE)
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
    plt.suptitle(f"{network.name} ($s_y$={sparsity['y'][-1]:.3f})")

    loss_x = compute_loss(x.T, labels)
    axes[0].axhline(y=loss_x, xmin=0, xmax=N_ITERS - 1, ls='--', color='gray', label="input 'x'")
    for name in ('h', 'y'):
        line = axes[0].plot(range(N_ITERS), loss[name], label=f"output '{name}'")[0]
        axes[1].plot(range(N_ITERS), accuracy[name], color=line.get_color())
        axes[2].plot(range(N_ITERS), convergence[name], color=line.get_color())
    axes[0].legend()

    results_dir = Path("results") / "clustering"
    results_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(results_dir / f"convergence {network.name}.png", dpi=300)

    fig, ax = plt.subplots()
    centroids = cluster_centroids(output_prev['y'].T, labels)
    im = ax.imshow(centroids, aspect='auto', interpolation='none')
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Class label")
    ax.set_title("Mean centroids of 'y'")
    plt.colorbar(im)
    fig.savefig(results_dir / f"centroids {network.name}.png", dpi=300)
    plt.show()
