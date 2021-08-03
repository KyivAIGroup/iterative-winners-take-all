# Similarity preservation

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from kwta import iWTA, kWTA

mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['savefig.format'] = 'png'

mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 14

np.random.seed(0)

KWTA_SPARSITY = 0.33

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


def cos(x1, x2):
    return (x1 @ x2.T) / np.sqrt(np.sum(x1) * np.sum(x2))


def generate_sample(x_c, distance):
    result = np.copy(x_c)
    idx_turn_off = np.random.choice(np.nonzero(x_c)[0], size=distance,
                                    replace=False)
    idx_turn_on = np.random.choice(np.nonzero(1 - x_c)[0], size=distance,
                                   replace=False)
    result[idx_turn_off] = 0
    result[idx_turn_on] = 1
    return result


iters = 20

def plot_cosine(a, a_x=20, kwta=False, color=None):
    print('new plot...')
    weights = {
        'w_xy': generate_random_matrix(N_y, N_x, a['xy']),
        'w_xh': generate_random_matrix(N_h, N_x, a['xh']),
        'w_hy': generate_random_matrix(N_y, N_h, a['hy']),
        'w_hh': generate_random_matrix(N_h, N_h, a['hh']),
        'w_yh': generate_random_matrix(N_h, N_y, a['yh']),
        'w_yy': generate_random_matrix(N_y, N_y, a['yy']),
    }

    distance_range = np.arange(a_x + 1)
    cos_x = np.zeros((a_x + 1, iters))
    cos_y = np.zeros((a_x + 1, iters))
    cos_y_kwta = np.zeros((a_x + 1, iters))
    sy_data = np.zeros((a_x + 1, iters))

    kwta_ky = int(KWTA_SPARSITY * N_y)

    for i, d in enumerate(distance_range):
        x_c = generate_random_vector(N_x, a_x)
        h_c, y_c = iWTA(x_c, **weights)
        if kwta:
            y_c_kwta = kWTA(weights['w_xy'] @ x_c, k=kwta_ky)
        for j in range(iters):
            xs = generate_sample(x_c, d)
            cos_x[i, j] = cos(x_c, xs)
            h, y = iWTA(xs, **weights)
            if kwta:
                y_kwta = kWTA(weights['w_xy'] @ xs, k=kwta_ky)
                cos_y_kwta[i, j] = cos(y_c_kwta, y_kwta)
            cos_y[i, j] = cos(y_c, y)
            sy_data[i, j] = np.count_nonzero(y) / N_y

    cos_x = np.mean(cos_x, axis=1)
    cos_y_mean = np.mean(cos_y, axis=1)
    cos_y_std = np.std(cos_y, axis=1)
    plt.plot(cos_x, cos_y_mean,
             label=rf'$s_y={np.mean(sy_data):.2f}$', color=color)
    plt.fill_between(cos_x, cos_y_mean + cos_y_std, cos_y_mean - cos_y_std,
                     alpha=0.2, color=color)

    if kwta:
        cos_y_mean = np.mean(cos_y_kwta, axis=1)
        cos_y_std = np.std(cos_y_kwta, axis=1)
        plt.plot(cos_x, cos_y_mean,
                 label=rf'kWTA, $s_y={KWTA_SPARSITY}$', color='#93003a')
        plt.fill_between(cos_x, cos_y_mean + cos_y_std, cos_y_mean - cos_y_std,
                         alpha=0.2, color='#93003a')


N_x = 100
N_y = 200
N_h = 200

# The no. of active synapses in a weight matrix per output neuron
a = {
    'xy': 50,
    'xh': 20,
    'hy': 20,
    'hh': 20,
    'yh': 20,
    'yy': 5,
}

plot_cosine(a, color='#00429d')

a['xy'] = 30
plot_cosine(a, kwta=True, color='#5681b9')


a['xy'] = 15
plot_cosine(a, color='#93c4d2')

plt.ylim([0, 1])
plt.xlim([0, 1])
plt.xlabel(r'$\cos(x_1, x_2)$')
plt.ylabel(r'$\cos(y_1, y_2)$')
plt.legend()
plt.title("Similarity preservation")
plt.savefig('figures/similarity_preservation')
plt.show()
