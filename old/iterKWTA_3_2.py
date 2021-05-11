# distribution of Z

import numpy as np
import matplotlib.pyplot as plt

N_x = 100
N_y = 500
s_w = 0.1
s_w_lat = 0.2
s_x = 0.3

# np.random.seed(1)

w = np.random.binomial(1, s_w, size=(N_y, N_x))
w_lat = np.random.binomial(1, s_w_lat, size=(N_y, N_y))


def kWTAi(v, w_lat, learning=False):
    y = np.zeros(v.size, dtype=int)
    Z = np.zeros(20)
    if learning:
        pass
    else:
        for ti in range(np.max(v), -1, -1):
            z = v - w_lat @ y >= ti
            # z = v  >= ti
            # print(np.count_nonzero(z), ti)
            y = np.logical_or(y, z)
            Z[ti] = np.count_nonzero(y) #- y @ z
            # Z[ti] = np.count_nonzero()
    # Z = -np.diff(Z, prepend=0)
    Z = -np.diff(Z, append=0)
    # Z[1] = 0 #, prepend=50)
    return y.astype(int), Z



# x2 = np.random.binomial(1, s_x, size=N_x)
# print('x overlap:', x @ x2.T)
# print(x, x2)

T = 1000
S = np.zeros([T, 10])
# y0 = kWTAi(w @ x, w_lat)
# y20 = kWTAi(w @ x2, w_lat)
# print('y overlap:', y0 @ y20.T)
Z = np.zeros((1, 20))
for i in range(1):
    x = np.random.binomial(1, s_x, size=N_x)
    y, Z[i] = kWTAi(w @ x, w_lat)

print(np.mean(Z, axis=0))
plt.bar(range(20), np.mean(Z, axis=0))
plt.show()
quit()
for t in range(T):
    y = kWTAi(w @ x, w_lat)
    y2 = kWTAi(w @ x2, w_lat)
    # print('y overlap:',t,  y @ y0.T)
    y_nonzero = np.nonzero(y)[0]
    group_connectivity = np.count_nonzero(w_lat[y_nonzero, y_nonzero])
    S[t, 0] = np.count_nonzero(y) / N_y
    S[t, 1] = group_connectivity / (np.count_nonzero(y) ** 2)
    S[t, 2] = np.count_nonzero(y2) / N_y
    S[t, 3] = np.count_nonzero(w_lat) / N_y ** 2
    S[t, 4] = y @ y0.T / np.count_nonzero(y)
    S[t, 5] = y @ (1 - y0.T) / np.count_nonzero(y)
    for _ in range(20):
        inds = np.random.choice(y_nonzero, 2)
        w_lat[inds[0], inds[1]] = 1

    # print(np.count_nonzero(w_lat))
    # print(inds, y_nonzero)
        # w_lat[np.random.choice(y_nonzero, 2)] = 1



plt.plot(range(T), S[:, 0], label='s_y')
plt.plot(range(T), S[:, 1], label='s_w(y)')
plt.plot(range(T), S[:, 2], label='s_y2')
plt.plot(range(T), S[:, 3], label='s_w')
plt.plot(range(T), S[:, 4], label='y dot y0')
plt.plot(range(T), S[:, 5], label='y dot not y0')
plt.ylim([0, 0.9])
plt.legend()
plt.show()
