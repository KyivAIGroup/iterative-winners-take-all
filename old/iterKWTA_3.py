# probability reflection

# repetition of the input sparsifies its representation

# why nonspecific decrease of sparsity set ups at strange attractor?

import numpy as np
import matplotlib.pyplot as plt

N_x = 100
N_y = 500
s_w = 0.1
s_w_lat = 0.2
s_x = 0.5

# np.random.seed(0)

w = np.random.binomial(1, s_w, size=(N_y, N_x))
w_lat = np.random.binomial(1, s_w_lat, size=(N_y, N_y))


def kWTAi(v, w_lat):
    y = np.zeros(v.size, dtype=int)
    for ti in range(np.max(v), 0, -1):
      z = v - w_lat @ y >= ti
      y = np.logical_or(y, z)
    return y


x = np.random.binomial(1, s_x, size=N_x)
x2 = np.random.binomial(1, s_x, size=N_x)
T = 1000
S = np.zeros([T, 4])
for t in range(T):
    y = kWTAi(w @ x, w_lat)
    y2 = kWTAi(w @ x2, w_lat)

    y_nonzero = np.nonzero(y)[0]
    group_connectivity = np.count_nonzero(w_lat[y_nonzero, y_nonzero])
    S[t, 0] = np.count_nonzero(y) / N_y
    S[t, 1] = group_connectivity / (np.count_nonzero(y) ** 2)
    S[t, 2] = np.count_nonzero(y2) / N_y
    S[t, 3] = np.count_nonzero(w_lat) / N_y ** 2
    for _ in range(200):
        inds = np.random.choice(y_nonzero, 2)
        w_lat[inds[0], inds[1]] = 1

    # print(np.count_nonzero(w_lat))
    # print(inds, y_nonzero)
        # w_lat[np.random.choice(y_nonzero, 2)] = 1



plt.plot(range(T), S[:, 0], label='group size')
plt.plot(range(T), S[:, 1], label='group connectivity')
plt.plot(range(T), S[:, 2], label='other group size')
plt.plot(range(T), S[:, 3], label='weights sparsity')
plt.ylim([0, 0.5])
plt.legend()
plt.show()

quit()
s_w_lat = np.linspace(0, 1, 20)

iters = 20
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
