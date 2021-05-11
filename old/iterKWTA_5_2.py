# excitory and inhibitory populations
# calculate average sparsities

# same connectivity, give same sparsity levels

# add self excitation

import numpy as np
import matplotlib.pyplot as plt

N_x = 100
s_x = 0.5
x = np.random.binomial(1, s_x, size=N_x)

def get_sparsity():
    N_y = 200
    N_h = 300
    s_w_xy = 0.1
    s_w_xh = 0.1
    w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))


    s_w_hh = 0.1
    s_w_hy = 0.1
    s_w_yh = 0.01
    s_w_yy = 0.01
    w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
    w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
    w_yy = np.random.binomial(1, s_w_yy, size=(N_y, N_y))
    w_yh = np.random.binomial(1, s_w_yh, size=(N_h, N_y))


    h_o = w_xh @ x
    y_o = w_xy @ x

    h = np.zeros(N_h, dtype=int)
    y = np.zeros(N_y, dtype=int)

    t_start = np.max([np.max(h_o), np.max(y_o)])
    for ti in range(t_start, 0, -1):
      z_h = h_o - w_hh @ h + w_yh @ y >= ti
      z_y = y_o - w_hy @ h + w_yy @ y >= ti
      h = np.logical_or(h, z_h)
      y = np.logical_or(y, z_y)

    a_y = np.count_nonzero(y)
    a_h = np.count_nonzero(h)
    return a_y, a_h

iters = 200
data = np.zeros((iters, 2))

for i in range(iters):
    data[i] = get_sparsity()

print(np.mean(data, axis=0))
quit()


def kWTAi(v, w_lat):
    y = np.zeros(v.size, dtype=int)
    for ti in range(np.max(v), 0, -1):
      z = v - w_lat @ y >= ti
      y = np.logical_or(y, z)
    return y

h = kWTAi(w_xh @ x, w_hh)


plt.plot(s_w_lat, np.mean(Y, axis=0) / N_y)
plt.xlabel(r'$s_{wlat}$, lateral weight sparsity')
plt.ylabel(r'$s_y$, output layer sparsity')
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.show()
