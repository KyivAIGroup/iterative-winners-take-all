# New learning rule
# gather stats
# try to change the weigth so to make permanence matrix equiporbable

# the weigth converge to stable!

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def update_weights(w, x_pre, x_post):
    # x_pre and x_post are vectors of shape (N, trials)
    assert x_pre.shape[1] == x_post.shape[1]
    for x, y in zip(x_pre.T, x_post.T):
        # outer product of all combinations
        x_pre_idx = x.nonzero()[0]
        x_post_idx = y.nonzero()[0]
        w[np.expand_dims(x_post_idx, axis=1), x_pre_idx] = 1


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


def kWTA2d(x, k):
    # x is a (N,) vec or (N, trials) tensor
    # print(x.shape)
    if k == 0:
        return np.zeros_like(x)
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    # print(x.shape)
    winners = np.argsort(x, axis=0)[-k:]  # (k, trials)
    sdr = np.zeros_like(x)
    sdr[winners, range(x.shape[1])] = 1
    return sdr.squeeze()


N_x = 50
N_y = 100

a_x = 10
a_w = 200
a_y = 20
s_x = a_x / N_x
s_y = a_y / N_y

N_data = 1000
# w_xy = generate_random_matrix(N_y, N_x, a_w)  # Note that a_w is NOT the no. of non-zero weights
w_xy = np.random.binomial(1, p=a_w / (N_y * N_x), size=(N_y, N_x))
X = generate_random_matrix(N_data, N_x, a_x)

Y_prev = None
w_xy_prev = w_xy.copy()
print(f"X sparsity: {X.mean()}")
print(f"w_xy sparsity: {w_xy.mean()}")

iters = 10
for i in range(iters):
    Y = kWTA2d(w_xy @ X.T, a_y)
    print()
    if Y_prev is not None:
        print(f"Y bits flipped after update (per sample): {(Y_prev ^ Y).sum(axis=1).mean()}")
    Y_prev = Y.copy()
    update_weights(w_xy, x_pre=X.T, x_post=Y)
    print(f"w_xy bits flipped after update: {(w_xy_prev ^ w_xy).sum()}")
    print(f"w_xy sparsity: {w_xy.mean()}")
    w_xy_prev = w_xy.copy()
