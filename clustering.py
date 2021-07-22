"""
Clustering experiment shows that if the data is sampled from noisy clusters,
the noise in the iWTA output is reduced after training.
"""
import numpy as np

from experiment import run_experiment
from networks import *

# Fix the random seed to reproduce the results
np.random.seed(0)

N_x = N_y = N_h = 200
s_x = 0.2

N_ITERS = 20
N_CHOOSE = None
LEARNING_RATE = 0.01

N_CLASSES = 10
N_SAMPLES_PER_CLASS = 100


x_centroids = np.random.binomial(1, s_x, size=(N_CLASSES, N_x))
labels = np.repeat(np.arange(N_CLASSES), N_SAMPLES_PER_CLASS)
np.random.shuffle(labels)
x = x_centroids[labels].T
white_noise = np.random.binomial(1, 0.5 * s_x, size=x.shape)
x ^= white_noise

for network_cls in (NetworkPermanenceVaryingSparsity,
                    NetworkWillshaw,
                    NetworkKWTA,
                    NetworkPermanenceFixedSparsity,
                    NetworkPermanenceVogels):
    run_experiment(x, labels, network_cls=network_cls,
                   weights_learn=('w_xy', 'w_xh', 'w_hy', 'w_hh', 'w_yh'),
                   n_iters=N_ITERS, n_choose=N_CHOOSE, lr=LEARNING_RATE,
                   experiment_name="clustering")
