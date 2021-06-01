import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from constants import RESULTS_DIR
from kwta import iWTA, update_weights, kWTA, kWTA_different_k
from graph import plot_assemblies
from utils import overlap

N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh, s_w_yy = 0.5, 0.1, 0.1, 0.1, 0.1, 0.02
N_REPEATS, N_ITERS = 10, 10
K_FIXED = int(0.15 * N_y)
INHIBIT_Y_OVERLAP = False

STATS_LABELS = "ovl($y_1^{noisy}, y_1$) - ovl($y_1, y_2$)", \
               "ovl($y_1^{noisy}, y_2$) - ovl($y_1, y_2$)", \
               r"ovl($y_1, y_2$)"
stats = {
    mode: np.zeros((N_REPEATS, N_ITERS, 3), dtype=np.float32)
    for mode in ('iWTA', 'kWTA', 'kWTA-fixed-k')
}
stats['nonzero'] = np.zeros((N_REPEATS, N_ITERS), dtype=np.float32)

def overlap2d(y_tensor):
    y1, y2, y1_noisy = y_tensor.T
    overlap_z = overlap(y1, y2)
    return overlap(y1_noisy, y1) - overlap_z, \
           overlap(y1_noisy, y2) - overlap_z, \
           overlap_z

def inhibit_overlap(tensor):
    z = tensor[:, 0] & tensor[:, 1]
    z = np.expand_dims(z, axis=1)  # (N, 1)
    tensor ^= z


for repeat in trange(N_REPEATS):
    x12 = np.random.binomial(1, s_x, size=(N_x, 2))
    x12_noisy = x12 + np.random.binomial(1, 0.05, size=x12.shape)
    x_stacked = np.c_[x12, x12_noisy[:, 0]]

    w_xy, w_xh, w_hy, w_hh, w_yy = {}, {}, {}, {}, {}
    for mode in stats.keys():
        w_xy[mode] = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
        w_xh[mode] = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
        w_hy[mode] = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
        w_hh[mode] = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
        w_yy[mode] = np.random.binomial(1, s_w_yy, size=(N_h, N_h))

    for iter_id in range(N_ITERS):
        h, y = {}, {}
        h['iWTA'], y['iWTA'] = iWTA(y0=w_xy['iWTA'] @ x_stacked,
                                    h0=w_xh['iWTA'] @ x_stacked,
                                    w_hy=w_hy['iWTA'],
                                    w_yy=w_yy['iWTA'])
        if INHIBIT_Y_OVERLAP:
            inhibit_overlap(y['iWTA'])

        stats['nonzero'][repeat, iter_id] = np.count_nonzero(y['iWTA'], axis=0).mean()

        for mode in ('kWTA', 'kWTA-fixed-k'):
            h[mode] = kWTA(w_xh[mode] @ x_stacked, k=K_FIXED)
            # h[mode] = kWTA_different_k(w_xh[mode] @ x_stacked, ks=np.count_nonzero(h['iWTA'], axis=0))
            y[mode] = w_xy[mode] @ x_stacked - w_hy[mode] @ h[mode]
        y['kWTA'] = kWTA_different_k(y['kWTA'], ks=np.count_nonzero(y['iWTA'], axis=0))
        y['kWTA-fixed-k'] = kWTA(y['kWTA-fixed-k'], k=K_FIXED)
        if INHIBIT_Y_OVERLAP:
            inhibit_overlap(y['kWTA'])
            inhibit_overlap(y['kWTA-fixed-k'])

        for mode in ('kWTA-fixed-k', 'kWTA', 'iWTA'):
            stats[mode][repeat, iter_id] = overlap2d(y[mode])
            update_weights(w_hy[mode], x_pre=h[mode][:, :2],
                           x_post=y[mode][:, :2], n_choose=10)
            # update_weights(w_hy[mode], x_pre=h[mode][:, learn_id],
            #                x_post=y[mode][:, 1 - learn_id], n_choose=5)
            update_weights(w_yy[mode], x_pre=y[mode][:, :2],
                           x_post=y[mode][:, :2], n_choose=5)

for mode in ('kWTA-fixed-k', 'kWTA', 'iWTA'):
    plot_assemblies(y[mode],
                    labels=('$y_1$', '$y_2$', '$y_1^{noisy}$'),
                    title=mode)
    plt.savefig(RESULTS_DIR / f"assembly_{mode}.png")

colormap = ['green', 'red', 'blue']

fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0)

axes[0].plot(range(N_ITERS), stats.pop('nonzero').mean(axis=0), ls='dashed',
             lw=2, label='nonzero', color='gray')

for key, ax in zip(stats.keys(), axes):
    mean = stats[key].mean(axis=0)
    std = stats[key].std(axis=0)
    for i, label in enumerate(STATS_LABELS):
        ax.plot(range(N_ITERS), mean[:, i], lw=2, label=label,
                color=colormap[i])
        ax.fill_between(range(N_ITERS), mean[:, i] + std[:, i],
                        mean[:, i] - std[:, i],
                        facecolor=colormap[i], alpha=0.1)
    ax.set_ylabel(key)
axes[0].legend(bbox_to_anchor=(1.02, 1))
axes[2].set_xlabel('Iteration')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "mean_of_two_classes.jpg")
