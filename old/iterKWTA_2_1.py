# simple version with one inhibitory neuron
# yes, it makes sparse encoding, and the input and weight sparsity slightly influence the s_y
# changing parameter alpha controls the spread of s_y

# how the alpha controls

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

def kWTAi_simple(v, alpha):
    y = np.zeros(v.size, dtype=int)
    for ti in range(np.max(v), 0, -1):
      z = v - alpha * np.count_nonzero(y) >= ti
      y = np.logical_or(y, z)
    return y


def plot_alpha(alpha):
    s_x_range = np.linspace(0.05, 0.95, 19)
    iters = 200
    Y = np.zeros((iters, s_x_range.size))
    for j, sxi in enumerate(s_x_range):
        for i in range(iters):
            x = np.random.binomial(1, sxi, size=N_x)
            y = kWTAi_simple(w @ x, alpha)
            Y[i, j] = np.count_nonzero(y)

    plt.plot(s_x_range, np.mean(Y, axis=0)/N_y, label=str(alpha))

plot_alpha(0.1)
plot_alpha(0.5)
plot_alpha(1)
plot_alpha(2)

plt.xlabel(r"s_x")
plt.ylabel(r"s_y")
plt.legend()
plt.show()
