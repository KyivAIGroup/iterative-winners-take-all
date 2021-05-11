# excitory and inhibitory populations
# calculate average sparsities

# same connectivity, give same sparsity levels

import numpy as np
import matplotlib.pyplot as plt

N_x = 100
s_x = 0.5
x = np.random.binomial(1, s_x, size=N_x)

def get_sparsity():
    N_y = 200
    N_h = 200
    s_w_xy = 0.1
    s_w_xh = 0.1
    w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))


    s_w_hh = 0.1
    s_w_hy = 0.1
    s_w_yh = 0.1
    s_w_yy = 0.1
    w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
    w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))


    h_o = w_xh @ x
    y_o = w_xy @ x

    h = np.zeros(N_h, dtype=int)
    y = np.zeros(N_y, dtype=int)

    t_start = np.max([np.max(h_o), np.max(y_o)])
    for ti in range(t_start, 0, -1):
      z_h = h_o - w_hh @ h >= ti
      z_y = y_o - w_hy @ h >= ti
      h = np.logical_or(h, z_h)
      y = np.logical_or(y, z_y)

    a_y = np.count_nonzero(y)
    a_h = np.count_nonzero(h)
    return a_y, a_h

iters = 200
data = np.zeros((iters, 2))

for i in range(iters):
    # a_y, a_h = get_sparsity()
    data[i] = get_sparsity()

print(np.mean(data, axis=0))
