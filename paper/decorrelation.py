"""
For a valid comparison with kWTA, only w_hy is learned.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import trange

from kwta import iWTA, update_weights, iWTA_history, kWTA
from permanence import PermanenceVogels, PermanenceFixedSparsity, \
    PermanenceVaryingSparsity

# Fix the random seed to reproduce the results
np.random.seed(0)

N_x = N_h = N_y = 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh = 0.5, 0.1, 0.1, 0.1, 0.1
N_REPEATS, N_ITERS = 20, 5
N_CHOOSE = 100


def pairwise_overlap(y):
    # y is a (N, samples) tensor
    y = y.T
    n_samples = y.shape[0]
    overlap = []
    for i in range(n_samples - 1):
        for j in range(i + 1, n_samples):
           ovl = (y[i] & y[j]).mean()
           overlap.append(ovl)
    return np.mean(overlap)


for method in (None, PermanenceFixedSparsity, PermanenceVogels, PermanenceVaryingSparsity, 'kWTA'):
    overlap_stats = np.zeros((N_REPEATS, N_ITERS), dtype=np.float32)
    weight_sparsity = {w: np.zeros(N_REPEATS) for w in ("w_hy", "w_hh")}

    for repeat in trange(N_REPEATS):
        x = np.random.binomial(1, s_x, size=(N_x, 10))

        w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
        w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
        w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
        w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))

        if method in (PermanenceFixedSparsity, PermanenceVogels, PermanenceVaryingSparsity):
            w_hy = method(w_hy, excitatory=False)

        for iter_id in range(N_ITERS):
            if method is PermanenceVogels:
                z_h, z_y = iWTA_history(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh)
                w_hy.update(x_pre=z_h, x_post=z_y, n_choose=N_CHOOSE)
                h, y = z_h[0], z_y[0]
                for i in range(1, len(z_h)):
                    h |= z_h[i]
                    y |= z_y[i]
            elif method in (PermanenceFixedSparsity, PermanenceVaryingSparsity):
                h, y = iWTA(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh)
                w_hy.update(x_pre=h, x_post=y, n_choose=N_CHOOSE)
            elif method is None:
                # classical Willshaw
                h, y = iWTA(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh)
                update_weights(w_hy, x_pre=h, x_post=y, n_choose=N_CHOOSE)
            else:
                # kWTA
                h = kWTA(w_xh @ x, k=10)
                y = kWTA(w_xy @ x - w_hy @ h, k=10)
                update_weights(w_hy, x_pre=h, x_post=y, n_choose=N_CHOOSE)
            overlap_stats[repeat, iter_id] = pairwise_overlap(y)
        weight_sparsity["w_hy"][repeat] = w_hy.mean()
        weight_sparsity["w_hh"][repeat] = w_hh.mean()

    for w, w_sparsity in weight_sparsity.items():
        print(f"{w} final sparsity: {w_sparsity.mean():.3f}")

    fig, ax = plt.subplots()

    mean = overlap_stats.mean(axis=0)
    std = overlap_stats.std(axis=0)

    ax.plot(range(N_ITERS), mean)
    ax.fill_between(range(N_ITERS), mean + std, mean - std, alpha=0.2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel("overlap($y_1$, $y_2$)")
    if method == 'kWTA':
        experiment_name = method
    elif method is None:
        experiment_name = 'willshaw'
    else:
        experiment_name = method.__name__
    ax.set_title(experiment_name)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f"decorrelation_{experiment_name}.jpg")
    plt.show()
