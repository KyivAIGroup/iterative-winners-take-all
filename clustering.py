import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import trange

from kwta import iWTA, update_weights, RESULTS_DIR, kWTA, kWTA_different_k

N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh, s_w_yy = 0.5, 0.1, 0.1, 0.1, 0.1, 0.02
N_REPEATS, N_ITERS = 10, 10
K_FIXED = int(0.1 * N_y)
NUM_TO_LEARN = 2
N_SAMPLES_PER_CLASS = 10

centroids = np.random.binomial(1, s_x, size=(N_x, 2))
centroids = centroids[:, centroids.any(axis=0)]
n_classes = centroids.shape[1]

xs = np.repeat(centroids, repeats=N_SAMPLES_PER_CLASS, axis=1)
labels = np.repeat(np.arange(n_classes), N_SAMPLES_PER_CLASS)
white_noise = np.random.binomial(1, 0.1, size=xs.shape)
xs ^= white_noise

stats = {
    mode: np.zeros((N_REPEATS, N_ITERS), dtype=np.float32)
    for mode in ('iWTA', 'kWTA', 'kWTA-fixed-k')
}
n_active = np.zeros((N_REPEATS, N_ITERS), dtype=np.float32)


def compute_discriminative_factor(tensor, labels):
    dist = pdist(tensor.T)
    dist = squareform(dist)
    factor = []
    for label in np.unique(labels):
        mask_same = labels == label
        dist_same = dist[mask_same][:, mask_same]
        dist_other = dist[mask_same][:, ~mask_same]
        dist_same = squareform(dist_same).mean()
        factor.append(dist_other / dist_same)
    factor = np.mean(factor)
    return factor


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

        y_kwta_pre = y0_batch - w_hy @ h0_batch
        _, y_batch = iWTA(y0=y0_batch, h0=h0_batch, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy)
        y_kwta_fixed_k = kWTA(y_kwta_pre, k=K_FIXED)
        n_active_batch = np.count_nonzero(y_batch, axis=0)
        y_kwta = kWTA_different_k(y_kwta_pre,
                                  ks=n_active_batch)
        n_active[experiment, epoch] = n_active_batch.mean()

        stats['iwta'][experiment, epoch] = compute_discriminative_factor(y_batch, labels)
        stats['kwta'][experiment, epoch] = compute_discriminative_factor(y_kwta, labels)
        stats['kwta-fixed-k'][experiment, epoch] = compute_discriminative_factor(y_kwta_fixed_k, labels)

colormap = {
    'iwta': 'green',
    'kwta': 'blue',
    'kwta-fixed-k': 'cyan'
}

fig, ax = plt.subplots()

n_active = n_active.mean(axis=0)
print(f"n_active={n_active}")
for key in stats.keys():
    mean = stats[key].mean(axis=0)
    std = stats[key].std(axis=0)
    ax.plot(range(N_ITERS), mean, lw=2, label=key, color=colormap[key])
    ax.fill_between(range(N_ITERS), mean + std, mean - std,
                    facecolor=colormap[key], alpha=0.3)
ax.set_title(f"Clustering. num_to_learn={NUM_TO_LEARN}")
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Discriminative factor (inner / intra dist)')
plt.savefig(RESULTS_DIR / "clustering.jpg")
plt.show()
