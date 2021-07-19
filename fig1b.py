# How weigth sparsity changes the encoding sparsity

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

# mpl.rcParams['grid.color'] = 'k'
# mpl.rcParams['grid.linestyle'] = ':'
# mpl.rcParams['grid.linewidth'] = 0.5

# mpl.rcParams['figure.figsize'] = [10.0, 8.0]
# mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 800
mpl.rcParams['savefig.format'] = 'eps'

mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.titlesize'] = 14

# np.random.seed(0)


def generate_random_vector(N, a_x, mask=None):
    vector = np.zeros(N, dtype=int)
    if mask is None:
        ones = np.random.choice(N, size=a_x, replace=False)
    else:
        ones = np.random.choice(np.nonzero(mask)[0], size=a_x, replace=False)
    vector[ones] = 1
    return vector

def generate_random_matrix(R, N, a_x):
    matrix = np.zeros((R, N), dtype=int)
    for i in range(R):
        matrix[i] = generate_random_vector(N, a_x)
    return matrix



def iWTA(x, w, N_y, N_h):
    h_o = w['xh'] @ x
    y_o = w['xy'] @ x

    h = np.zeros(N_h, dtype=int)
    y = np.zeros(N_y, dtype=int)

    t_start = np.max([np.max(h_o), np.max(y_o)])
    Z_h = []
    for ti in range(t_start, 0, -1):
      z_h = h_o - w['hh'] @ h + w['yh'] @ y >= ti
      z_y = y_o - w['hy'] @ h + w['yy'] @ y >= ti
      h = np.logical_or(h, z_h)
      y = np.logical_or(y, z_y)
      Z_h.append(z_h.astype(int))

    return Z_h



N_x = 200
N_y = 200
N_h = 200

a_x = 50



a = {
    'xy': 0,
    'xh': 20,
    'hy': 0,
    'hh': 10,
    'yh': 0,
    'yy': 0,
}
w = {
    'xy': generate_random_matrix(N_y, N_x, a['xy']),
    'xh': generate_random_matrix(N_h, N_x, a['xh']),
    'hy': generate_random_matrix(N_y, N_h, a['hy']),
    'hh': generate_random_matrix(N_h, N_h, a['hh']),
    'yh': generate_random_matrix(N_h, N_y, a['yh']),
    'yy': generate_random_matrix(N_y, N_y, a['yy']),
     }
#
#
x = generate_random_vector(N_x, a_x)
Z_h = np.array(iWTA(x, w, N_y, N_h)).T

num_to_plot = 20
start = 100
t_range = Z_h.shape[1]
im = plt.imshow(Z_h[start:start + num_to_plot], cmap='binary',
                interpolation='none', vmin=0, vmax=1, aspect='equal')

ax = plt.gca()

ax.set_xticks(np.arange(-.5, t_range, 1), minor=False)
ax.set_yticks(np.arange(-.5, num_to_plot, 1), minor=False)

ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_xlabel(r'$t_n$, interation')
ax.set_ylabel(r'$z^h$, neuron number')
ax.grid(which='major', color='k', linestyle='-', linewidth=1)
# ax.grid(True)
# plt.axis('off')
# plt.savefig('figures/fig1b', bbox_inches='tight')
plt.show()