# simple version with one inhibitory neuron
# yes, it makes sparse encoding, and the input and weight sparsity slightly influence the s_y
# changing parameter alpha controls the spread of s_y

import numpy as np
import matplotlib.pyplot as plt

N_x = 100
N_y = 500
s_w = 0.5
s_w_lat = 0.1
s_x = 0.8
x = np.random.binomial(1, s_x, size=N_x)
w = np.random.binomial(1, s_w, size=(N_y, N_x))
w_lat = np.random.binomial(1, s_w_lat, size=(N_y, N_y))

alpha = 0.2

def kWTAi_simple(v):
    y = np.zeros(v.size, dtype=int)
    for ti in range(np.max(v), 0, -1):
      z = v - alpha * np.count_nonzero(y) >= ti
      y = np.logical_or(y, z)
    return y

s_x_range = np.linspace(0.05, 0.95, 19)
# print(s_x_range)
iters = 200
Y = np.zeros((iters, s_x_range.size))
for j, sxi in enumerate(s_x_range):
    for i in range(iters):
        x = np.random.binomial(1, sxi, size=N_x)
        y = kWTAi_simple(w @ x)
        Y[i, j] = np.count_nonzero(y)

plt.plot(s_x_range, np.mean(Y, axis=0)/N_y, label='sx')


x = np.random.binomial(1, 0.5, size=N_x)
s_x_range = np.linspace(0.05, 0.95, 19)
iters = 200
Y = np.zeros((iters, s_x_range.size))
for j, swi in enumerate(s_x_range):
    for i in range(iters):

        w = np.random.binomial(1, swi, size=(N_y, N_x))
        y = kWTAi_simple(w @ x)
        Y[i, j] = np.count_nonzero(y)

plt.plot(s_x_range, np.mean(Y, axis=0)/N_y, label='sw')
plt.xlabel(r"s_w and s_w")
plt.ylabel(r"s_y")
plt.legend()
plt.show()
quit()
s_w_lat = np.linspace(0, 1, 20)

iters = 5
Y = np.zeros((iters, s_w_lat.size))
for iter in range(iters):
  print(iter)
  for k, s_wi in enumerate(s_w_lat):
    w_lat = np.random.binomial(1, s_wi, size=(N_y, N_y))
    Y[iter, k] = np.count_nonzero(kWTAi(w @ x, w_lat))


plt.plot(s_w_lat, np.mean(Y, axis=0) / N_y)
plt.xlabel(r'$s_{wlat}$, lateral weight sparsity')
plt.ylabel(r'$s_y$, output layer sparsity')
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.show()
