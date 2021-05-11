import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import trange

from kwta import kWTAi, update_weights, RESULTS_DIR, kWTA, kWTA_different_k

N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy = 0.5, 0.1, 0.1, 0.1
N_repeat, N_epoch = 10, 10
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
    mode: np.zeros((N_repeat, N_epoch), dtype=np.float32)
    for mode in ('iwta', 'kwta', 'kwta-fixed-k')
}
n_active = np.zeros((N_repeat, N_epoch), dtype=np.float32)


def compute_discriminative_factor(tensor, labels):
    dist = pdist(tensor.T)
    dist = squareform(dist)
    factor = []
    for label in np.unique(labels):
        mask_same = labels == label
        dist_inner = dist[mask_same][:, mask_same]
        dist_intra = dist[~mask_same][:, ~mask_same]
        dist_inner = squareform(dist_inner).mean()
        dist_intra = squareform(dist_intra).mean()
        factor.append(dist_intra / dist_inner)
    factor = np.mean(factor)
    return factor


for experiment in trange(N_repeat):
    w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))

    for epoch in range(N_epoch):
        shuffle = np.arange(xs.shape[1])
        np.random.shuffle(shuffle)
        xs = xs[:, shuffle]
        labels = labels[shuffle]

        y0_batch = w_xy @ xs
        h0_batch = w_xh @ xs
        for y0, h0 in zip(y0_batch.T, h0_batch.T):
            h, y = kWTAi(y0=y0, h0=h0, w_hy=w_hy)
            update_weights(w_hy, x_pre=h, x_post=y, n_choose=NUM_TO_LEARN)

        y_kwta_pre = y0_batch - w_hy @ h0_batch
        _, y_batch = kWTAi(y0=y0_batch, h0=h0_batch, w_hy=w_hy)
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
print(f"n_active={n_active.astype(int)}")
for key in stats.keys():
    mean = stats[key].mean(axis=0)
    std = stats[key].std(axis=0)
    ax.plot(range(N_epoch), mean, lw=2, label=key, color=colormap[key])
    ax.fill_between(range(N_epoch), mean + std, mean - std,
                    facecolor=colormap[key], alpha=0.3)
ax.set_title(f"Clustering. num_to_learn={NUM_TO_LEARN}")
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Discriminative factor (intra / inner dist)')
plt.savefig(RESULTS_DIR / "clustering.jpg")
plt.show()
