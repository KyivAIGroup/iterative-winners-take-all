# excitory and inhibitory populations

import numpy as np
import matplotlib.pyplot as plt

N_x = 100
s_x = 0.5
x = np.random.binomial(1, s_x, size=N_x)


N_y = 200
N_h = 200
s_w = 0.1
w_xy = np.random.binomial(1, s_w, size=(N_y, N_x))
w_xh = np.random.binomial(1, s_w, size=(N_h, N_x))


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
  # print("thresh", ti)
  # print(np.count_nonzero(h))
  # print(np.count_nonzero(y))


print(np.count_nonzero(h))
print(np.count_nonzero(y))
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
