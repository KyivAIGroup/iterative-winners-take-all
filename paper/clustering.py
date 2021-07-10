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
from utils import compute_clustering_coefficient

# Fix the random seed to reproduce the results
np.random.seed(0)

N_x = N_y = N_h = 200
s_x = 0.2
s_w_xh = s_w_xy = s_w_hy = s_w_yy = s_w_hh = s_w_yh = 0.05

N_REPEATS, N_ITERS = 1, 50
N_CHOOSE = 100

N_CLASSES = 10
N_SAMPLES_PER_CLASS = 100


def compute_accuracy(output, labels):
    mean = np.vstack([output[labels == l].mean(0) for l in np.unique(labels)])
    output = output / np.linalg.norm(output, axis=1, keepdims=True)
    mean = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    similarity = output.dot(mean.T)  # (n_samples, n_classes)
    labels_pred = similarity.argmax(axis=1)
    accuracy = np.mean(labels == labels_pred)
    return accuracy


for method in (None, PermanenceFixedSparsity, PermanenceVaryingSparsity, 'kWTA'):
    y_sparsity = np.zeros((N_REPEATS, N_ITERS))
    clustering_coefficient = np.zeros((N_REPEATS, N_ITERS))
    accuracy = np.zeros((N_REPEATS, N_ITERS))
    accuracy_x = np.zeros(N_REPEATS)
    weight_sparsity = {w: np.zeros(N_REPEATS) for w in ("w_hy", "w_hh", "w_yh", "w_yy")}

    for repeat in trange(N_REPEATS):

        centroids = np.random.binomial(1, s_x, size=(N_CLASSES, N_x))
        labels = np.repeat(np.arange(N_CLASSES), N_SAMPLES_PER_CLASS)
        np.random.shuffle(labels)
        x = centroids[labels].T
        white_noise = np.random.binomial(1, 0.5 * s_x, size=x.shape)
        x ^= white_noise

        accuracy_x[repeat] = compute_accuracy(x.T, labels)

        w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
        w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
        w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
        w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
        w_yh = np.random.binomial(1, s_w_yh, size=(N_h, N_y))
        w_yy = np.random.binomial(1, s_w_yy, size=(N_y, N_y))

        if method in (PermanenceFixedSparsity, PermanenceVogels, PermanenceVaryingSparsity):
            w_hy = method(w_hy, excitatory=False)
            w_hh = method(w_hh, excitatory=False)
            w_yh = method(w_yh, excitatory=True)
            w_yy = method(w_yy, excitatory=True)

        for iter_id in trange(N_ITERS):
            if method is PermanenceVogels:
                z_h, z_y = iWTA_history(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh, w_yh=w_yh, w_yy=w_yy)
                w_hy.update(x_pre=z_h, x_post=z_y, n_choose=N_CHOOSE)
                w_hh.update(x_pre=z_h, x_post=z_h, n_choose=N_CHOOSE)
                w_yh.update(x_pre=z_y, x_post=z_h, n_choose=N_CHOOSE)
                w_yy.update(x_pre=z_y, x_post=z_y, n_choose=N_CHOOSE)
                h, y = z_h[0], z_y[0]
                for i in range(1, len(z_h)):
                    h |= z_h[i]
                    y |= z_y[i]
            elif method in (PermanenceFixedSparsity, PermanenceVaryingSparsity):
                h, y = iWTA(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh, w_yh=w_yh, w_yy=w_yy)
                w_hy.update(x_pre=h, x_post=y, n_choose=N_CHOOSE)
                w_hh.update(x_pre=h, x_post=h, n_choose=N_CHOOSE)
                w_yh.update(x_pre=y, x_post=h, n_choose=N_CHOOSE)
                w_yy.update(x_pre=y, x_post=y, n_choose=N_CHOOSE)
            elif method is None:
                # classical Willshaw
                h, y = iWTA(x, w_xh=w_xh, w_xy=w_xy, w_hy=w_hy, w_hh=w_hh, w_yh=w_yh, w_yy=w_yy)
                update_weights(w_hy, x_pre=h, x_post=y, n_choose=N_CHOOSE)
                update_weights(w_hh, x_pre=h, x_post=h, n_choose=N_CHOOSE)
                update_weights(w_yh, x_pre=y, x_post=h, n_choose=N_CHOOSE)
                update_weights(w_yy, x_pre=y, x_post=y, n_choose=N_CHOOSE)
            else:
                # kWTA
                h = kWTA(w_xh @ x, k=10)
                y = kWTA(w_xy @ x - w_hy @ h, k=10)
                update_weights(w_hy, x_pre=h, x_post=y, n_choose=N_CHOOSE)
                update_weights(w_hy, x_pre=h, x_post=y, n_choose=N_CHOOSE)
                update_weights(w_hh, x_pre=h, x_post=h, n_choose=N_CHOOSE)
                update_weights(w_yh, x_pre=y, x_post=h, n_choose=N_CHOOSE)
                update_weights(w_yy, x_pre=y, x_post=y, n_choose=N_CHOOSE)
            y_sparsity[repeat, iter_id] = y.mean()
            clustering_coefficient[repeat, iter_id] = compute_clustering_coefficient(y.T, labels)
            accuracy[repeat, iter_id] = compute_accuracy(y.T, labels)
        weight_sparsity["w_hy"][repeat] = w_hy.mean()
        weight_sparsity["w_hh"][repeat] = w_hh.mean()
        weight_sparsity["w_yh"][repeat] = w_yh.mean()
        weight_sparsity["w_yy"][repeat] = w_yy.mean()

    accuracy_x = np.mean(accuracy_x)
    print(f"{accuracy_x=}")
    y_sparsity = y_sparsity.mean(axis=0)
    print(f"{y_sparsity=}")

    for w, w_sparsity in weight_sparsity.items():
        print(f"{w} final sparsity: {w_sparsity.mean():.3f}")

    fig, axes = plt.subplots(nrows=2, sharex=True)
    axes[0].set_ylabel("Clustering coefficient")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Iteration")
    if method == 'kWTA':
        experiment_name = method
    elif method is None:
        experiment_name = 'Willshaw'
    else:
        experiment_name = method.__name__
    experiment_name = f"{experiment_name} ($s_y$ = {y_sparsity[-1]:.3f})"
    plt.suptitle(experiment_name)

    for ax, metric in zip(axes, [clustering_coefficient, accuracy]):
        mean = metric.mean(axis=0)
        std = metric.std(axis=0)
        ax.plot(range(N_ITERS), mean)
        ax.fill_between(range(N_ITERS), mean + std, mean - std, alpha=0.2)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f"clustering_{experiment_name}.jpg")
    plt.show()
