import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import trange

from kwta import iWTA, update_weights, iWTA_history
from permanence import PermanenceVogels, PermanenceFixedSparsity, \
    PermanenceVaryingSparsity
from utils import generate_k_active

# Fix the random seed to reproduce the results
np.random.seed(0)

N_x = N_h = N_y = 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh = 0.2, 0.1, 0.1, 0.1, 0.1
N_REPEATS, N_ITERS = 10, 5
N_CHOOSE = 10


def sample_from_distribution(px, n_neurons, n_samples, k):
    px = np.array(px)
    assert np.isclose(px.sum(), 1), "Probabilities must sum up to 1"
    x = []
    labels = []
    for i, pxi in enumerate(px):
        repeats = math.ceil(pxi * n_samples)
        xi = generate_k_active(n=n_neurons, k=k)
        xi = np.tile(xi, reps=(repeats, 1))
        x.append(xi)
        labels_i = np.full(repeats, fill_value=i)
        labels.append(labels_i)
    x = np.vstack(x)
    labels = np.hstack(labels)
    # Shuffling is not required; we do this to illustrate that the obtained
    # results are not due to the sequential nature of the input data.
    shuffle = np.random.permutation(len(labels))
    x = x[shuffle]
    labels = labels[shuffle]
    x = x.T
    return x, labels


px = [0.1, 0.1, 0.8]

for perm_cls in (None, PermanenceFixedSparsity, PermanenceVogels, PermanenceVaryingSparsity):
    y_sparsity = np.zeros((N_REPEATS, N_ITERS, len(px)), dtype=np.float32)

    for repeat in trange(N_REPEATS):
        x, labels = sample_from_distribution(px=px, n_neurons=N_x, n_samples=30, k=10)

        w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
        w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
        w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
        w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))

        if perm_cls is not None:
            w_hy = perm_cls(w_hy, excitatory=False)

        for iter_id in range(N_ITERS):
            if perm_cls is PermanenceVogels:
                z_h, z_y = iWTA_history(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh)
                w_hy.update(x_pre=z_h, x_post=z_y, n_choose=N_CHOOSE)
                h, y = z_h[0], z_y[0]
                for i in range(1, len(z_h)):
                    h |= z_h[i]
                    y |= z_y[i]
            elif perm_cls in (PermanenceFixedSparsity, PermanenceVaryingSparsity):
                h, y = iWTA(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh)
                w_hy.update(x_pre=h, x_post=y, n_choose=N_CHOOSE)
            else:
                # classical Willshaw
                h, y = iWTA(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh)
                update_weights(w_hy, x_pre=h, x_post=y, n_choose=N_CHOOSE)
            y_sparsity_i = np.count_nonzero(y, axis=0).astype(float) / N_y
            for label in range(len(px)):
                mask = labels == label
                y_sparsity[repeat, iter_id, label] = y_sparsity_i[mask].mean()

    fig, ax = plt.subplots()

    mean = y_sparsity.mean(axis=0)
    std = y_sparsity.std(axis=0)

    for label, (m, s) in enumerate(zip(mean.T, std.T)):
        ax.plot(range(N_ITERS), m, label=f"$x_{label}$")
        ax.fill_between(range(N_ITERS), m + s, m - s, alpha=0.2)
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel("$s_y$")
    experiment_name = "willshaw" if perm_cls is None else perm_cls.__name__
    ax.set_title(experiment_name)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f"habituation_{experiment_name}.jpg")
    plt.show()
