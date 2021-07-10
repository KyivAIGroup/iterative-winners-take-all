import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import trange

from kwta import iWTA, update_weights, iWTA_history
from permanence import PermanenceVogels, PermanenceFixedSparsity, \
    PermanenceVaryingSparsity

# Fix the random seed to reproduce the results
np.random.seed(0)

N_x = N_h = N_y = 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh = 0.2, 0.1, 0.1, 0.1, 0.1
N_REPEATS, N_ITERS = 10, 50
N_CHOOSE = 10
N_SAMPLES_TOTAL = 30


def generate_k_active(n, k):
    x = np.zeros(n, dtype=np.int32)
    active = np.random.choice(n, size=k, replace=False)
    x[active] = 1
    return x


def sample_from_distribution(px, n_neurons, n_samples, k):
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


px = [0.2, 0.2, 0.6]

for method in (None, PermanenceFixedSparsity, PermanenceVogels, PermanenceVaryingSparsity):
    method = PermanenceVaryingSparsity
    y_sparsity = np.zeros((N_REPEATS, N_ITERS, len(px)), dtype=np.float32)
    weight_sparsity = {w: np.zeros(N_REPEATS) for w in ("w_hy", "w_hh")}

    for repeat in trange(N_REPEATS):
        x, labels = sample_from_distribution(px=px, n_neurons=N_x, n_samples=N_SAMPLES_TOTAL, k=int(s_x * N_x))

        w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
        w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
        w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
        w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))

        if method is not None:
            w_hy = method(w_hy, excitatory=False)
            w_hh = method(w_hh, excitatory=False)

        for iter_id in range(N_ITERS):
            if method is PermanenceVogels:
                z_h, z_y = iWTA_history(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh)
                w_hy.update(x_pre=z_h, x_post=z_y, n_choose=N_CHOOSE)
                # w_hh.update(x_pre=z_h, x_post=z_h, n_choose=N_CHOOSE)
                h, y = z_h[0], z_y[0]
                for i in range(1, len(z_h)):
                    h |= z_h[i]
                    y |= z_y[i]
            elif method in (PermanenceFixedSparsity, PermanenceVaryingSparsity):
                h, y = iWTA(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh)
                w_hy.update(x_pre=h, x_post=y, n_choose=N_CHOOSE)
                # w_hh.update(x_pre=h, x_post=h, n_choose=N_CHOOSE)
            else:
                # classical Willshaw
                h, y = iWTA(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh)
                update_weights(w_hy, x_pre=h, x_post=y, n_choose=N_CHOOSE)
                # update_weights(w_hh, x_pre=h, x_post=h, n_choose=N_CHOOSE)
            y_sparsity_i = y.mean(axis=0)
            for label in range(len(px)):
                mask = labels == label
                y_sparsity[repeat, iter_id, label] = y_sparsity_i[mask].mean()
        weight_sparsity["w_hy"][repeat] = w_hy.mean()
        weight_sparsity["w_hh"][repeat] = w_hh.mean()

    for w, w_sparsity in weight_sparsity.items():
        print(f"{w} final sparsity: {w_sparsity.mean():.3f}")

    fig, ax = plt.subplots()

    mean = y_sparsity.mean(axis=0)
    std = y_sparsity.std(axis=0)

    for label, (m, s) in enumerate(zip(mean.T, std.T)):
        ax.plot(range(N_ITERS), m, label=f"$x_{label}$")
        ax.fill_between(range(N_ITERS), m + s, m - s, alpha=0.2)
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel("$s_y$")
    experiment_name = "willshaw" if method is None else method.__name__
    ax.set_title(experiment_name)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f"habituation_{experiment_name}.jpg")
    plt.show()
