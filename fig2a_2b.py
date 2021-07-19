"""
How weight sparsity changes the encoding sparsity.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from kwta import iWTA

# mpl.rcParams['grid.color'] = 'k'
# mpl.rcParams['grid.linestyle'] = ':'
# mpl.rcParams['grid.linewidth'] = 0.5

# mpl.rcParams['figure.figsize'] = [10.0, 8.0]
# mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 800
mpl.rcParams['savefig.format'] = 'pdf'

mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.titlesize'] = 14


def generate_random_vector(N, a_x):
    """
    Generate a binary vector of size `N` with exactly `a_x` active neurons.
    """
    vector = np.zeros(N, dtype=int)
    ones = np.random.choice(N, size=a_x, replace=False)
    vector[ones] = 1
    return vector


def generate_random_matrix(R, N, a_x):
    """
    Generate a binary matrix of size (R, N) with exactly `a_x` active neurons
    per row.
    """
    matrix = np.zeros((R, N), dtype=int)
    for i in range(R):
        matrix[i] = generate_random_vector(N, a_x)
    return matrix

N_x = 100
N_y = 100
N_h = 100

a_x = 10

# The number of active neurons in the output vector
a = {
    'xy': 20,
    'xh': 20,
    'hy': 20,
    'hh': 20,
    'yh': 20,
    'yy': 5,
}
w = {
    'w_xy': generate_random_matrix(N_y, N_x, a['xy']),
    'w_xh': generate_random_matrix(N_h, N_x, a['xh']),
    'w_hy': generate_random_matrix(N_y, N_h, a['hy']),
    'w_hh': generate_random_matrix(N_h, N_h, a['hh']),
    'w_yh': generate_random_matrix(N_h, N_y, a['yh']),
    'w_yy': generate_random_matrix(N_y, N_y, a['yy']),
}

iters = 10


def plot_w(weight, s='y'):
    weight = f"w_{weight}"
    print(f'plotting {weight}')
    N = w[weight].shape[1]
    w_range = np.arange(1, N, 5)
    Y = np.zeros((w_range.size, iters, N_y))
    H = np.zeros((w_range.size, iters, N_h))
    s_y = np.zeros((w_range.size, iters))
    s_h = np.zeros((w_range.size, iters))
    for k, a_i in enumerate(w_range):
        w[weight] = generate_random_matrix(w[weight].shape[0],
                                           w[weight].shape[1], a_i)
        for i in range(iters):
            x = generate_random_vector(N_x, a_x)
            H[k, i], Y[k, i] = iWTA(x, **w)
            s_y[k, i] = np.mean(Y[k, i])
            s_h[k, i] = np.mean(H[k, i])

    if s == 'y':
        s_y_mean = np.mean(s_y, axis=1)
        s_y_std = np.std(s_y, axis=1)
        ax.plot(w_range / N, s_y_mean, label='w$^{%s}$' % weight[2:])
        ax.fill_between(w_range / N, s_y_mean + s_y_std, s_y_mean - s_y_std,
                        alpha=0.5)

    if s == 'h':
        s_h_mean = np.mean(s_h, axis=1)
        s_h_std = np.std(s_h, axis=1)
        ax.plot(w_range / N, s_h_mean, label='w$^{%s}$' % weight[2:])
        ax.fill_between(w_range / N, s_h_mean + s_h_std, s_h_mean - s_h_std,
                        alpha=0.5)
    # return to default
    w[weight] = generate_random_matrix(w[weight].shape[0], w[weight].shape[1],
                                       a[weight[2:]])


fig, ax = plt.subplots(1)
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

plot_w('xy')
plot_w('xh')
plot_w('hy')
plot_w('hh')
plot_w('yh')
plot_w('yy')

ax.legend()
ax.set_xlabel(r'$s_w$, weights sparsity')
ax.set_ylabel(r'$s_y$, y layer sparsity')
plt.ylim([0, 1.05])
plt.xlim([0, 1])
fig.savefig(results_dir / 'fig2a.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1)
plot_w('xy', 'h')
plot_w('xh', 'h')
plot_w('hy', 'h')
plot_w('hh', 'h')
plot_w('yh', 'h')
plot_w('yy', 'h')

ax.legend()
ax.set_xlabel(r'$s_w$, weights sparsity')
ax.set_ylabel(r'$s_h$, h layer sparsity')
plt.ylim([0, 1.05])
plt.xlim([0, 1])
fig.savefig(results_dir / 'fig2b.pdf', bbox_inches='tight')
plt.show()
