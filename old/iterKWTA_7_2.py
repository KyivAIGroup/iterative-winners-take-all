# excitory and inhibitory populations
# Similarity preservation

# compare iWTA to kWTA

# kwta uses the same sparsity as iWTA
# the encodings quite different (cos(y1, y2) around 0.6 dependent on s_x

import numpy as np
import matplotlib.pyplot as plt


# np.random.seed(0)


N_x = 100
s_x = 0.5
x = np.random.binomial(1, s_x, size=N_x)

N_y = 200
N_h = 200
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

def kWTA(cells, sparsity):
    n_active = max(int(sparsity * cells.size), 1)
    winners = np.argsort(cells)[-n_active:]
    sdr = np.zeros(cells.shape, dtype=cells.dtype)
    sdr[winners] = 1
    return sdr


def activate(x, learning=False):
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

    if learning:
        num_to_learn = 0
        inds_pre = np.random.choice(np.nonzero(h)[0], num_to_learn)
        inds_post = np.random.choice(np.nonzero(y)[0], num_to_learn)
        w_hy[inds_post, inds_pre] = 1

        num_to_learn = 0
        inds_pre = np.random.choice(np.nonzero(y)[0], num_to_learn)
        inds_post = np.random.choice(np.nonzero(y)[0], num_to_learn)
        w_yy[inds_post, inds_pre] = 1

        num_to_learn = 0
        inds_pre = np.random.choice(np.nonzero(h)[0], num_to_learn)
        inds_post = np.random.choice(np.nonzero(h)[0], num_to_learn)
        w_hh[inds_post, inds_pre] = 1

        num_to_learn = 2
        inds_pre = np.random.choice(np.nonzero(y)[0], num_to_learn)
        inds_post = np.random.choice(np.nonzero(h)[0], num_to_learn)
        w_yh[inds_post, inds_pre] = 1

    # a_y = np.count_nonzero(y)
    # a_h = np.count_nonzero(h)

    return y.astype(int), h.astype(int)

def cos(x1, x2):
    return (x1 @ x2.T) / np.sqrt(np.sum(x1) * np.sum(x2))

s_x = 0.1
# x1 = np.random.binomial(1, s_x, size=N_x)
# x2 = np.random.binomial(1, s_x, size=N_x)
# print('before learning')
# y1, h1 = activate(x1)
# y2, h2 = activate(x2)


# print(np.count_nonzero(y1), np.count_nonzero(y2))
# print(cos(x1, x2), cos(y1, y2))

iters = 200

# s_y = 0.1
data = np.zeros((iters, 3))
data_kwta = np.zeros((iters, 2))
sparsity = np.zeros((iters, 2))
for i in range(iters):
    x1 = np.random.binomial(1, s_x, size=N_x)
    x2 = np.random.binomial(1, s_x, size=N_x)
    y1, h1 = activate(x1)
    y2, h2 = activate(x2)
    s_y1 = np.count_nonzero(y1) / N_y
    s_y2 = np.count_nonzero(y2) / N_y
    y_kwta1 = kWTA(w_xy @ x1, s_y1)
    y_kwta2 = kWTA(w_xy @ x2, s_y2)
    data[i] = cos(x1, x2), cos(y1, y_kwta1), cos(y2, y_kwta2)
    data_kwta[i] = y1 @ y_kwta1, y2 @ y_kwta2
    sparsity[i] = np.count_nonzero(y1), np.count_nonzero(y2)

# print(data)
print(np.mean(data, axis=0))
print(np.mean(data_kwta, axis=0))
print(np.mean(sparsity, axis=0))
# print(np.var(data, axis=0))











quit()
print(activate(x1))
print(activate(x2))
for _ in range(200):
    activate(x1, learning=True)
print('after learning')
print(activate(x1))
print(activate(x2))



quit()
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
