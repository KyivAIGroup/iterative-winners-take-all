# show how the latteral weight sparsity defines the y sparsity

# add the influence of the input sparsity

# plot sy vs sx  (the dependence is linear)

import numpy as np
import matplotlib.pyplot as plt

N_x = 100
N_y = 500
s_w = 0.5

w = np.random.binomial(1, s_w, size=(N_y, N_x))



def kWTAi(v, w_lat):
    y = np.zeros(v.size, dtype=int)
    for ti in range(np.max(v), 0, -1):
      z = v - w_lat @ y >= ti
      y = np.logical_or(y, z)
    return y






def plot_sw(s_w_lat):
  iters = 50
  s_x = np.linspace(0.05, 0.95, 20)
  w_lat = np.random.binomial(1, s_w_lat, size=(N_y, N_y))

  Y = np.zeros((iters, s_x.size))
  for iter in range(iters):
    print(iter)
    for k, s_xi in enumerate(s_x):
      x = np.random.binomial(1, s_xi, size=N_x)
      Y[iter, k] = np.count_nonzero(kWTAi(w @ x, w_lat))
  plt.plot(s_x, np.mean(Y, axis=0) / N_y, label=str(s_w_lat))

plot_sw(0.1)
plot_sw(0.5)
plot_sw(0.8)


plt.xlabel(r'$s_{x}$, input sparsity')
plt.ylabel(r'$s_y$, output layer sparsity')
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.legend()
plt.show()
