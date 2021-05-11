# show how the latteral weight sparsity defines the y sparsity

# add the influence of the input weight sparsity

# The same as for s_x. the rise of s_w slightly rises the swlat-sy curve

import numpy as np
import matplotlib.pyplot as plt

N_x = 100
N_y = 500
s_w = 0.1
s_w_lat = 0.1
s_x = 0.2
x = np.random.binomial(1, s_x, size=N_x)

w = np.random.binomial(1, s_w, size=(N_y, N_x))
w_lat = np.random.binomial(1, s_w_lat, size=(N_y, N_y))


def kWTAi(v, w_lat):
    y = np.zeros(v.size, dtype=int)
    for ti in range(np.max(v), 0, -1):
      z = v - w_lat @ y >= ti
      y = np.logical_or(y, z)
    return y

s_w_lat = np.linspace(0, 1, 20)

iters = 20

s_w = 0.1
Y = np.zeros((iters, s_w_lat.size))
for iter in range(iters):
  print(iter)
  w = np.random.binomial(1, s_w, size=(N_y, N_x))
  for k, s_wi in enumerate(s_w_lat):

    w_lat = np.random.binomial(1, s_wi, size=(N_y, N_y))
    Y[iter, k] = np.count_nonzero(kWTAi(w @ x, w_lat))
plt.plot(s_w_lat, np.mean(Y, axis=0) / N_y, label=str(s_w))


s_w = 0.4
Y = np.zeros((iters, s_w_lat.size))
for iter in range(iters):
  print(iter)
  w = np.random.binomial(1, s_w, size=(N_y, N_x))
  for k, s_wi in enumerate(s_w_lat):
    w_lat = np.random.binomial(1, s_wi, size=(N_y, N_y))
    Y[iter, k] = np.count_nonzero(kWTAi(w @ x, w_lat))
plt.plot(s_w_lat, np.mean(Y, axis=0) / N_y, label=str(s_w))

s_w = 0.8
Y = np.zeros((iters, s_w_lat.size))
for iter in range(iters):
  print(iter)
  w = np.random.binomial(1, s_w, size=(N_y, N_x))
  for k, s_wi in enumerate(s_w_lat):
    w_lat = np.random.binomial(1, s_wi, size=(N_y, N_y))
    Y[iter, k] = np.count_nonzero(kWTAi(w @ x, w_lat))
plt.plot(s_w_lat, np.mean(Y, axis=0) / N_y, label=str(s_w))


plt.xlabel(r'$s_{wlat}$, lateral weight sparsity')
plt.ylabel(r'$s_y$, output layer sparsity')
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.legend()
plt.show()
