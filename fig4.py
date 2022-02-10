"""
Clustering experiment shows that if the data is sampled from noisy clusters,
the noise in the iWTA output is reduced after training.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from experiment import run_experiment, COLORS
from networks import *

# Fix the random seed to reproduce the results
np.random.seed(0)

# The dimensionality of input vector 'x' and output populations 'h' and 'y'
N_x = N_h = N_y = 200

# The sparsity of input vectors 'x'
s_x = 0.2

# Repeat the experiment N times
N_REPEATS = 5

# The no. of full iterations to run
N_ITERS = 20

# N_CHOOSE defines the number of synapses to update from a sample pair.
# It controls how much the boolean matrix 'm' is filled.
# Set to None to update all active synapses.
N_CHOOSE = None

# The learning rate
LEARNING_RATE = 0.01

# Generate 10 random clusters with 100 samples each
N_CLASSES = 10
N_SAMPLES_PER_CLASS = 100


x_centroids = np.random.binomial(1, s_x, size=(N_CLASSES, N_x))
labels = np.repeat(np.arange(N_CLASSES), N_SAMPLES_PER_CLASS)
np.random.shuffle(labels)
x = x_centroids[labels].T
white_noise = np.random.binomial(1, 0.5 * s_x, size=x.shape)
x ^= white_noise

metrics = dict()

for network_cls in (NetworkPermanenceVaryingSparsity,
                    NetworkSimpleHebb,
                    NetworkKWTA,
                    NetworkPermanenceFixedSparsity,
                    NetworkPermanenceVogels):
    network, metrics[network_cls] = run_experiment(
        x, labels,
        network_cls=network_cls,
        weights_learn=('w_xy', 'w_xh', 'w_hy', 'w_hh', 'w_yh'),
        n_iters=N_ITERS,
        n_choose=N_CHOOSE,
        lr=LEARNING_RATE,
        experiment_name="clustering")


# Plot iWTA vs kWTA comparison
fig, ax = plt.subplots()
ax.plot(range(N_ITERS), metrics[NetworkPermanenceVaryingSparsity]['error']['y'], label='iWTA', color=COLORS[0])
ax.plot(range(N_ITERS), metrics[NetworkKWTA]['error']['y'], label='kWTA', color=COLORS[1])
ax.set_xlim(xmin=0)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.xlabel("Epoch (~10 iterations)")
plt.ylabel(r"Error of the output  $y$  signal")
plt.title("Comparison of iWTA and kWTA on a clustering task", size=13)
plt.tight_layout()
plt.savefig("results/clustering/comparison.png",
            bbox_inches='tight')
plt.show()
