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
mpl.rcParams['savefig.format'] = 'pdf'

mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.titlesize'] = 14

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
    for ti in range(t_start, 0, -1):
      z_h = h_o - w['hh'] @ h + w['yh'] @ y >= ti
      z_y = y_o - w['hy'] @ h + w['yy'] @ y >= ti
      h = np.logical_or(h, z_h)
      y = np.logical_or(y, z_y)

    return y.astype(int), h.astype(int)


def iWTA2(x, w, N_y, N_h):
    h_o = w['xh'] @ x
    y_o = w['xy'] @ x

    h = np.zeros(N_h, dtype=int)
    y = np.zeros(N_y, dtype=int)

    t_start = np.max([np.max(h_o), np.max(y_o)])
    for ti in range(t_start, 0, -1):
      z_h = h_o - w['hh'] @ h + w['yh'] @ y >= ti
      z_y = y_o - w['hy'] @ h + w['yy'] @ y >= ti
      h = np.logical_or(h, z_h)
      y = np.logical_or(y, z_y)

    return y.astype(int), h.astype(int)

N_x = 100
N_y = 100
N_h = 100

a_x = 10

a = {
    'xy': 20,
    'xh': 20,
    'hy': 20,
    'hh': 20,
    'yh': 20,
    'yy': 5,
}
w = {
    'xy': generate_random_matrix(N_y, N_x, a['xy']),
    'xh': generate_random_matrix(N_h, N_x, a['xh']),
    'hy': generate_random_matrix(N_y, N_h, a['hy']),
    'hh': generate_random_matrix(N_h, N_h, a['hh']),
    'yh': generate_random_matrix(N_h, N_y, a['yh']),
    'yy': generate_random_matrix(N_y, N_y, a['yy']),
     }

iters = 10
def plot_w(weight, s='y'):
    print(f'plotting {weight}')
    N = w[weight].shape[1]
    w_range = np.arange(1, N, 5)
    Y = np.zeros((w_range.size, iters, N_y))
    H = np.zeros((w_range.size, iters, N_h))
    s_y = np.zeros((w_range.size, iters))
    s_h = np.zeros((w_range.size, iters))
    for k, a_i in enumerate(w_range):
        w[weight] = generate_random_matrix(w[weight].shape[0], w[weight].shape[1], a_i)
        for i in range(iters):
            x = generate_random_vector(N_x, a_x)
            Y[k, i], H[k, i] = iWTA(x, w, N_y, N_h)
            s_y[k, i] = np.mean(Y[k, i])
            s_h[k, i] = np.mean(H[k, i])

    if s == 'y':
        s_y_mean = np.mean(s_y, axis=1)
        s_y_std = np.std(s_y, axis=1)
        ax.plot(w_range / N, s_y_mean, label='w_' + weight)
        ax.fill_between(w_range / N, s_y_mean + s_y_std, s_y_mean - s_y_std, alpha=0.5)

    if s == 'h':
        s_h_mean = np.mean(s_h, axis=1)
        s_h_std = np.std(s_h, axis=1)
        ax.plot(w_range / N, s_h_mean, label='w_' + weight)
        ax.fill_between(w_range / N, s_h_mean + s_h_std, s_h_mean - s_h_std, alpha=0.5)
    # return to default
    w[weight] = generate_random_matrix(w[weight].shape[0], w[weight].shape[1], a[weight])


fig, ax = plt.subplots(1)

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
# plt.savefig('figures/fig2a', bbox_inches='tight')
plt.show()

quit()
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
# plt.savefig('figures/fig2b', bbox_inches='tight')
plt.show()