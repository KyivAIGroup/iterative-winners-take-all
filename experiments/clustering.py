import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from constants import RESULTS_DIR
from kwta import iWTA, update_weights, kWTA
from utils import compute_loss

N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh, s_w_yy = 0.5, 0.1, 0.1, 0.1, 0.1, 0.02
N_REPEATS, N_ITERS = 10, 10
K_FIXED = int(0.1 * N_y)
NUM_TO_LEARN = 2
N_SAMPLES_PER_CLASS = 10

centroids = np.random.binomial(1, s_x, size=(N_x, 2))
assert centroids.any(axis=0).all(), "Pick another seed"
n_classes = centroids.shape[1]

xs = np.repeat(centroids, repeats=N_SAMPLES_PER_CLASS, axis=1)
labels = np.repeat(np.arange(n_classes), N_SAMPLES_PER_CLASS)
white_noise = np.random.binomial(1, 0.1, size=xs.shape)
xs ^= white_noise

stats = {
    mode: np.zeros((N_REPEATS, N_ITERS), dtype=np.float32)
    for mode in ('iWTA', 'kWTA')
}
n_active = np.zeros((N_REPEATS, N_ITERS), dtype=np.float32)


for experiment in trange(N_REPEATS):
    w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
    w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
    w_yy = np.random.binomial(1, s_w_yy, size=(N_y, N_y))

    for epoch in range(N_ITERS):
        shuffle = np.arange(xs.shape[1])
        np.random.shuffle(shuffle)
        xs = xs[:, shuffle]
        labels = labels[shuffle]

        y0_batch = w_xy @ xs
        h0_batch = w_xh @ xs
        for y0, h0 in zip(y0_batch.T, h0_batch.T):
            h, y = iWTA(y0=y0, h0=h0, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy)
            update_weights(w_hy, x_pre=h, x_post=y, n_choose=NUM_TO_LEARN)
            update_weights(w_yy, x_pre=y, x_post=y, n_choose=NUM_TO_LEARN)

        _, y_batch = iWTA(y0=y0_batch, h0=h0_batch, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy)
        y_kwta = kWTA(y0_batch - w_hy @ kWTA(h0_batch, k=K_FIXED), k=K_FIXED)
        n_active[experiment, epoch] = np.count_nonzero(y_batch, axis=0).mean()

        stats['iWTA'][experiment, epoch] = compute_loss(y_batch.T, labels)
        stats['kWTA'][experiment, epoch] = compute_loss(y_kwta.T, labels)


fig, ax = plt.subplots()

n_active = n_active.mean(axis=0)
print(f"n_active={n_active}")
for key in stats.keys():
    mean = stats[key].mean(axis=0)
    std = stats[key].std(axis=0)
    line = ax.plot(range(N_ITERS), mean, lw=2, label=key)[0]
    ax.fill_between(range(N_ITERS), mean + std, mean - std,
                    facecolor=line.get_color(), alpha=0.3)
ax.set_title(f"num_to_learn={NUM_TO_LEARN}")
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.savefig(RESULTS_DIR / "clustering.jpg")
plt.show()
