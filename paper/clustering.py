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
from utils import compute_loss

# Fix the random seed to reproduce the results
np.random.seed(0)

N_x = N_y = N_h = 200
s_x = 0.2
s_w_xh = s_w_xy = s_w_hy = s_w_yy = s_w_hh = s_w_yh = 0.05

N_REPEATS, N_ITERS = 1, 5
N_CHOOSE = 100
LEARNING_RATE = 0.001

N_CLASSES = 10
N_SAMPLES_PER_CLASS = 10


for method in (ParameterBinary, PermanenceFixedSparsity, PermanenceVaryingSparsity, 'kWTA'):
    y_sparsity = np.zeros((N_REPEATS, N_ITERS))
    loss = np.zeros((N_REPEATS, N_ITERS))
    loss_x = np.zeros(N_REPEATS)
    weight_sparsity = {w: np.zeros(N_REPEATS) for w in ("w_hy", "w_hh", "w_yh", "w_yy")}

    for repeat in trange(N_REPEATS):

        centroids = np.random.binomial(1, s_x, size=(N_CLASSES, N_x))
        labels = np.repeat(np.arange(N_CLASSES), N_SAMPLES_PER_CLASS)
        np.random.shuffle(labels)
        x = centroids[labels].T
        white_noise = np.random.binomial(1, 0.5 * s_x, size=x.shape)
        x ^= white_noise

        loss_x[repeat] = compute_loss(x.T, labels)

        w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
        w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
        w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
        w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
        w_yh = np.random.binomial(1, s_w_yh, size=(N_h, N_y))
        w_yy = np.random.binomial(1, s_w_yy, size=(N_y, N_y))

        method_cls = ParameterBinary if method == 'kWTA' else method
        w_hy = method_cls(w_hy, excitatory=False)
        w_hh = method_cls(w_hh, excitatory=False)
        w_yh = method_cls(w_yh, excitatory=True)
        w_yy = method_cls(w_yy, excitatory=True)

        for iter_id in trange(N_ITERS):
            if method == 'kWTA':
                h = kWTA(w_xh @ x, k=10)
                y = kWTA(w_xy @ x - w_hy @ h, k=10)
                w_hy.update(x_pre=h, x_post=y, n_choose=N_CHOOSE)
            elif method is PermanenceVogels:
                z_h, z_y = iWTA_history(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh, w_yh=w_yh, w_yy=w_yy)
                w_hy.update(x_pre=z_h, x_post=z_y, n_choose=N_CHOOSE)
                w_hh.update(x_pre=z_h, x_post=z_h, n_choose=N_CHOOSE)
                w_yh.update(x_pre=z_y, x_post=z_h, n_choose=N_CHOOSE)
                w_yy.update(x_pre=z_y, x_post=z_y, n_choose=N_CHOOSE)
                h, y = z_h[0], z_y[0]
                for i in range(1, len(z_h)):
                    h |= z_h[i]
                    y |= z_y[i]
            else:
                h, y = iWTA(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh, w_yh=w_yh, w_yy=w_yy)
                w_hy.update(x_pre=h, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
                w_hh.update(x_pre=h, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
                w_yh.update(x_pre=y, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
                w_yy.update(x_pre=y, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)

            y_sparsity[repeat, iter_id] = y.mean()
            loss[repeat, iter_id] = compute_loss(y.T, labels)
        weight_sparsity["w_hy"][repeat] = w_hy.mean()
        weight_sparsity["w_hh"][repeat] = w_hh.mean()
        weight_sparsity["w_yh"][repeat] = w_yh.mean()
        weight_sparsity["w_yy"][repeat] = w_yy.mean()

    y_sparsity = y_sparsity.mean(axis=0)
    print(f"{y_sparsity=}")

    for w, w_sparsity in weight_sparsity.items():
        print(f"{w} final sparsity: {w_sparsity.mean():.3f}")

    fig, ax = plt.subplots()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    if method == 'kWTA':
        experiment_name = method
    else:
        experiment_name = method.__name__
    ax.set_title(f"{experiment_name} ($s_y$ = {y_sparsity[-1]:.3f})")

    mean = loss.mean(axis=0)
    std = loss.std(axis=0)
    ax.plot(range(N_ITERS), mean, label="Output Y")
    ax.fill_between(range(N_ITERS), mean + std, mean - std, alpha=0.2)
    loss_x = np.mean(loss_x)
    ax.axhline(y=loss_x, xmin=0, xmax=N_ITERS - 1, ls='--', color='gray', label="Input X")
    ax.legend()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f"clustering_{experiment_name}.jpg")
    plt.show()
