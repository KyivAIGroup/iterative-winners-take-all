"""
Decorrelation experiment shows that the initial overlap in input data 'x'
is decreased due to the w_hy weights that inhibit the overlapped area strongly
than non-overlapped areas.
"""

import numpy as np

from experiment import run_experiment
from networks import *

# Fix the random seed to reproduce the results
np.random.seed(0)

N_x = N_y = N_h = 200
s_x = 0.05

N_ITERS = 10
N_CHOOSE = None
LEARNING_RATE = 0.001


def generate_similar_input(x, n_generate, overlap=0.5):
    """
    Generate `n_generate` vectors with the same size and the number of active
    neurons in `x`. The overlap defines the similarity.

    Parameters
    ----------
    x : (N,) np.ndarray
        The input vector
    n_generate : int
        The number of similar vectors to generate.
    overlap : float
        How much the output vectors are similar to `x`.

    Returns
    -------
    x_similar : (N, n_generate) np.ndarray
        Random vectors that are similar to `x`.
    """
    idx_pool = x.nonzero()[0]
    no_overlap_idx = (x == 0).nonzero()[0]
    k = len(idx_pool)
    x_similar = np.zeros((len(x), n_generate), dtype=np.int32)
    k_common = int(overlap * k)
    for i in range(n_generate):
        active = np.random.choice(idx_pool, size=k_common, replace=False)
        active_no_overlap = np.random.choice(no_overlap_idx, size=k - k_common,
                                             replace=False)
        active = np.append(active, active_no_overlap)
        x_similar[active, i] = 1
    assert (x_similar.sum(axis=0) == k).all()
    return x_similar


x = np.random.binomial(1, s_x, size=N_x)
x = generate_similar_input(x, n_generate=100)
labels = np.arange(x.shape[1])
print(f"input 'x' sparsity: {x.mean()}")

for network_cls in (NetworkPermanenceVaryingSparsity,
                    NetworkWillshaw,
                    NetworkKWTA,
                    NetworkPermanenceFixedSparsity,
                    NetworkPermanenceVogels):
    run_experiment(x, labels, network_cls=network_cls,
                   architecture=('w_xy', 'w_xh', 'w_hy', 'w_hh'),
                   weights_learn=('w_hy',),
                   n_iters=N_ITERS, n_choose=N_CHOOSE, lr=LEARNING_RATE,
                   experiment_name="decorrelation")
