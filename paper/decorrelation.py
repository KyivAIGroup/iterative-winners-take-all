import numpy as np

from experiment import run_experiment
from networks import *

# Fix the random seed to reproduce the results
np.random.seed(0)

N_x = N_y = N_h = 200
s_x = 0.2

N_ITERS = 70
N_CHOOSE = 5
LEARNING_RATE = 0.01

x = np.random.binomial(1, s_x, size=(N_x, 100))
labels = np.arange(x.shape[1])

for network_cls in (NetworkPermanenceVaryingSparsity,
                    NetworkWillshaw,
                    NetworkKWTA,
                    NetworkPermanenceFixedSparsity,
                    NetworkPermanenceVogels):
    run_experiment(x, labels, network_cls=network_cls,
                   n_iters=N_ITERS, n_choose=N_CHOOSE, lr=LEARNING_RATE,
                   experiment_name="decorrelation")
