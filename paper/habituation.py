import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from pathlib import Path

from kwta import iWTA, update_weights, Permanence
from utils import generate_k_active

# Fix the random seed to reproduce the results
np.random.seed(0)

N_x = N_h = N_y = 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh = 0.2, 0.1, 0.1, 0.1, 0.1
N_REPEATS, N_ITERS = 10, 5


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

stats = {
    mode: np.zeros((N_REPEATS, N_ITERS, len(px)), dtype=np.int32)
    for mode in ('overlap', 'nonzero_count')
}

for repeat in trange(10):
    x, labels = sample_from_distribution(px=px, n_neurons=N_x, n_samples=30, k=10)

    w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
    w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))

    # w_xy = Permanence(np.random.rand(N_y, N_x), sparsity=s_w_xy)
    # w_xh = Permanence(np.random.rand(N_h, N_x), sparsity=s_w_xh)
    w_hy = Permanence(np.random.rand(N_y, N_h), sparsity=s_w_hy)
    # w_hh = Permanence(np.random.rand(N_h, N_h), sparsity=s_w_hh)

    _, y0 = iWTA(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh)

    for iter_id in range(N_ITERS):
        h, y = iWTA(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh)
        overlap = (y & y0).sum(axis=0)
        nonzero_count = y.sum(axis=0)
        for label in range(len(px)):
            mask = labels == label
            stats['overlap'][repeat, iter_id, label] = overlap[mask].mean()
            stats['nonzero_count'][repeat, iter_id, label] = nonzero_count[mask].mean()
        w_hy.update(x_pre=h, x_post=y, n_choose=None)
        # update_weights(w_hy, x_pre=h, x_post=y, n_choose=10)
    print("sparsity w_hy: ", w_hy.mean())

fig, ax = plt.subplots()

mean = stats["nonzero_count"].mean(axis=0)
std = stats["nonzero_count"].std(axis=0)
mean_overlap = stats["overlap"].mean(axis=0)

for label, (m, s) in enumerate(zip(mean.T, std.T)):
    line = ax.plot(range(N_ITERS), m, label=f"$x_{label}$")[0]
    ax.plot(range(N_ITERS), mean_overlap[:, label], lw=1, ls='--', color=line.get_color())
    ax.fill_between(range(N_ITERS), m + s, m - s, alpha=0.2)
ax.legend()
ax.set_xlabel('Iteration')
ax.set_ylabel("$||y||_0$")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
plt.savefig(results_dir / "habituation.jpg")
plt.show()
