# How  the encoding sparsity depends on the input sparsity for different N_y

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from kwta import iWTA

mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['savefig.format'] = 'png'

mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 14

np.random.seed(0)


def generate_random_vector(N, a_x):
    vector = np.zeros(N, dtype=int)
    ones = np.random.choice(N, size=a_x, replace=False)
    vector[ones] = 1
    return vector


def generate_random_matrix(R, N, a_x):
    matrix = np.zeros((R, N), dtype=int)
    for i in range(R):
        matrix[i] = generate_random_vector(N, a_x)
    return matrix


N_x = 100
N_y = 200
N_h = 200

# The no. of active synapses in a weight matrix per output neuron
a = {
    'xy': 20,
    'xh': 20,
    'hy': 20,
    'hh': 20,
    'yh': 20,
    'yy': 10,
}

# Increase 'iters' to make the figure smooth
iters = 10


def plot_dependence_on_input_sparsity(N_x, N_h, N_y, colors):
    weights = {
        'w_xy': generate_random_matrix(N_y, N_x, a['xy']),
        'w_xh': generate_random_matrix(N_h, N_x, a['xh']),
        'w_hy': generate_random_matrix(N_y, N_h, a['hy']),
        'w_hh': generate_random_matrix(N_h, N_h, a['hh']),
        'w_yh': generate_random_matrix(N_h, N_y, a['yh']),
        'w_yy': generate_random_matrix(N_y, N_y, a['yy']),
    }
    a_x_range = np.arange(1, N_x, int(N_x * 0.01))
    s_y = np.zeros((iters, a_x_range.size))
    s_h = np.zeros((iters, a_x_range.size))
    for i in trange(iters,
                    desc=f"Plotting dependence on s_x: N_h={N_h}, N_y={N_y}"):
        for k, ax_i in enumerate(a_x_range):
            x = generate_random_vector(N_x, ax_i)
            h, y = iWTA(x, **weights)
            s_y[i, k] = np.mean(y)
            s_h[i, k] = np.mean(h)

    s_y_mean = np.mean(s_y, axis=0)
    s_h_mean = np.mean(s_h, axis=0)
    s_y_std = np.std(s_y, axis=0)
    s_h_std = np.std(s_h, axis=0)

    ax.plot(a_x_range / N_x, s_h_mean, label=f'$s_h, N_y=N_h={N_y}$',
            color=colors[0])
    ax.fill_between(a_x_range / N_x, s_h_mean + s_h_std, s_h_mean - s_h_std,
                    alpha=0.2, color=colors[0])
    ax.plot(a_x_range / N_x, s_y_mean, label=f'$s_y, N_y=N_h={N_y}$',
            color=colors[1])
    ax.fill_between(a_x_range / N_x, s_y_mean + s_y_std, s_y_mean - s_y_std,
                    alpha=0.2, color=colors[1])


fig, ax = plt.subplots()

plot_dependence_on_input_sparsity(N_x=100, N_h=100, N_y=100,
                                  colors=['#73a2c6', '#f4777f'])
plot_dependence_on_input_sparsity(N_x=100, N_h=300, N_y=300,
                                  colors=['#00429d', '#93003a'])

handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax.legend(handles, labels)

ax.set_xlabel(r'$s_x$, input sparsity')
ax.set_ylabel('encoding sparsity')
plt.ylim([0, 1.05])
plt.xlim([0, 1])
plt.title("Dependence on the input sparsity")
plt.savefig('figures/fig2c', bbox_inches='tight')
plt.show()
