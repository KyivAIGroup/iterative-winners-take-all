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
from metrics import compute_loss, compute_accuracy

# Fix the random seed to reproduce the results
np.random.seed(0)

N_x = N_y = N_h = 200
s_x = 0.2
s_w_xh = s_w_xy = s_w_hy = s_w_yy = s_w_hh = s_w_yh = 0.05

N_ITERS = 5
N_CHOOSE = 100
LEARNING_RATE = 0.001

N_CLASSES = 10
N_SAMPLES_PER_CLASS = 10


def run_classical_willshaw(x, labels, weights: dict):
    for name, w in weights.items():
        weights[name] = ParameterBinary(w)

    sparsity = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    loss = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    accuracy = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    for iter_id in trange(N_ITERS, desc="classical Willshaw"):
        h, y = iWTA(x, **weights)
        weights['w_xy'].update(x_pre=x, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_xh'].update(x_pre=x, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_hh'].update(x_pre=h, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_hy'].update(x_pre=h, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_yy'].update(x_pre=y, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_yh'].update(x_pre=y, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        for name, output in zip(('h', 'y'), (h, y)):
            sparsity[name][iter_id] = output.mean()
            loss[name][iter_id] = compute_loss(output.T, labels)
            accuracy[name][iter_id] = compute_accuracy(output.T, labels)
    return loss, accuracy, sparsity


def run_permanence_fixed_sparsity(x, labels, weights: dict):
    for name, w in weights.items():
        weights[name] = PermanenceFixedSparsity(w)

    sparsity = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    loss = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    accuracy = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    for iter_id in trange(N_ITERS, desc="Permanence fixed sparsity"):
        h, y = iWTA(x, **weights)
        weights['w_xy'].update(x_pre=x, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_xh'].update(x_pre=x, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_hh'].update(x_pre=h, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_hy'].update(x_pre=h, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_yy'].update(x_pre=y, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_yh'].update(x_pre=y, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        for name, output in zip(('h', 'y'), (h, y)):
            sparsity[name][iter_id] = output.mean()
            loss[name][iter_id] = compute_loss(output.T, labels)
            accuracy[name][iter_id] = compute_accuracy(output.T, labels)
    return loss, accuracy, sparsity


def run_permanence_varying_sparsity(x, labels, weights: dict, output_sparsity_desired=(0.025, 0.1)):
    weights['w_xh'] = PermanenceVaryingSparsity(weights['w_xh'], excitatory=True, output_sparsity_desired=output_sparsity_desired)
    weights['w_xy'] = PermanenceVaryingSparsity(weights['w_xy'], excitatory=True, output_sparsity_desired=output_sparsity_desired)
    weights['w_hh'] = PermanenceVaryingSparsity(weights['w_hh'], excitatory=False, output_sparsity_desired=output_sparsity_desired)
    weights['w_hy'] = PermanenceVaryingSparsity(weights['w_hh'], excitatory=False, output_sparsity_desired=output_sparsity_desired)
    weights['w_yy'] = PermanenceVaryingSparsity(weights['w_yy'], excitatory=True, output_sparsity_desired=output_sparsity_desired)
    weights['w_yh'] = PermanenceVaryingSparsity(weights['w_yh'], excitatory=True, output_sparsity_desired=output_sparsity_desired)

    sparsity = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    loss = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    accuracy = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    for iter_id in trange(N_ITERS, desc="Permanence varying sparsity"):
        h, y = iWTA(x, **weights)
        weights['w_xy'].update(x_pre=x, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_xh'].update(x_pre=x, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_hh'].update(x_pre=h, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_hy'].update(x_pre=h, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_yy'].update(x_pre=y, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_yh'].update(x_pre=y, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        for name, output in zip(('h', 'y'), (h, y)):
            sparsity[name][iter_id] = output.mean()
            loss[name][iter_id] = compute_loss(output.T, labels)
            accuracy[name][iter_id] = compute_accuracy(output.T, labels)
    return loss, accuracy, sparsity


def run_permanence_vogels(x, labels, weights: dict):
    for name in ("w_xy", "w_xh", "w_yy", "w_yh"):
        weights[name] = PermanenceFixedSparsity(weights[name])
    for name in ("w_hy", "w_hh"):
        weights[name] = PermanenceVogels(weights[name])

    sparsity = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    loss = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    accuracy = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    for iter_id in trange(N_ITERS, desc="Permanence Vogels"):
        z_h, z_y = iWTA_history(x, **weights)
        h, y = z_h[0], z_y[0]
        for i in range(1, len(z_h)):
            h |= z_h[i]
            y |= z_y[i]

        weights['w_xy'].update(x_pre=x, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_xh'].update(x_pre=x, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_hh'].update(x_pre=z_h, x_post=z_h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_hy'].update(x_pre=z_h, x_post=z_y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_yy'].update(x_pre=y, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_yh'].update(x_pre=y, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)

        for name, output in zip(('h', 'y'), (h, y)):
            sparsity[name][iter_id] = output.mean()
            loss[name][iter_id] = compute_loss(output.T, labels)
            accuracy[name][iter_id] = compute_accuracy(output.T, labels)
    return loss, accuracy, sparsity


def run_kwta(x, labels, weights: dict):
    for name in ("w_xy", "w_xh", "w_hy"):
        weights[name] = ParameterBinary(w)

    sparsity = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    loss = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    accuracy = dict(y=np.zeros(N_ITERS), h=np.zeros(N_ITERS))
    for iter_id in trange(N_ITERS):
        h = kWTA(weights['w_xh'] @ x, k=10)
        y = kWTA(weights['w_xy'] @ x - weights['w_hy'] @ h, k=10)
        weights['w_xy'].update(x_pre=x, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_xh'].update(x_pre=x, x_post=h, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        weights['w_hy'].update(x_pre=h, x_post=y, n_choose=N_CHOOSE, lr=LEARNING_RATE)
        for name, output in zip(('h', 'y'), (h, y)):
            sparsity[name][iter_id] = output.mean()
            loss[name][iter_id] = compute_loss(output.T, labels)
            accuracy[name][iter_id] = compute_accuracy(output.T, labels)
    return loss, accuracy, sparsity


for method in (run_classical_willshaw, run_kwta, run_permanence_fixed_sparsity, run_permanence_varying_sparsity, run_permanence_vogels):
    centroids = np.random.binomial(1, s_x, size=(N_CLASSES, N_x))
    labels = np.repeat(np.arange(N_CLASSES), N_SAMPLES_PER_CLASS)
    np.random.shuffle(labels)
    x = centroids[labels].T
    white_noise = np.random.binomial(1, 0.5 * s_x, size=x.shape)
    x ^= white_noise

    print(f"'x' accuracy: {compute_accuracy(x.T, labels)}")

    weights = {}
    weights['w_xy'] = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    weights['w_xh'] = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    weights['w_hy'] = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
    weights['w_hh'] = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
    weights['w_yh'] = np.random.binomial(1, s_w_yh, size=(N_h, N_y))
    weights['w_yy'] = np.random.binomial(1, s_w_yy, size=(N_y, N_y))

    loss, accuracy, sparsity = method(x, labels, weights)
    for name in ('h', 'y'):
        print(f"'{name}' final sparsity: {sparsity[name][-1]}")
    for name, w in weights.items():
        print(f"{name} final sparsity: {w.mean():.3f}")

    fig, axes = plt.subplots(nrows=2, sharex=True)
    axes[1].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")
    if method == 'kWTA':
        experiment_name = method
    else:
        experiment_name = method.__name__
    plt.suptitle(f"{experiment_name} ($s_y$ = {sparsity['y'][-1]:.3f})")

    loss_x = compute_loss(x.T, labels)
    axes[0].axhline(y=loss_x, xmin=0, xmax=N_ITERS - 1, ls='--', color='gray', label="Input 'x'")
    for name in ('h', 'y'):
        line = axes[0].plot(range(N_ITERS), loss[name], label=f"Output '{name}'")[0]
        axes[1].plot(range(N_ITERS), accuracy[name], color=line.get_color())
    axes[0].legend()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f"clustering_{experiment_name}.jpg")
    plt.show()
