"""
Let y1 = f(x1, W) and y2 = f(x2, W).
Learning the weights either for (x1, y1) or (x2, y2) should decorrelate y1 and y2 signals.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from constants import RESULTS_DIR
from kwta import kWTA, iWTA, update_weights, kWTA_different_k
from utils import overlap

N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh, s_w_yy = 0.5, 0.1, 0.1, 0.1, 0.05, 0.05
N_REPEATS, N_ITERS = 10, 10
K_FIXED = int(0.15 * N_y)
NUM_TO_LEARN = 50

stats = {mode: np.zeros((N_REPEATS, N_ITERS), dtype=np.float32)
         for mode in ('kWTA-fixed-k', 'kWTA', 'iWTA', 'nonzero')}


def overlap2d(tensor):
    assert tensor.shape[1] == 2
    return overlap(tensor[:, 0], tensor[:, 1])


for repeat in trange(N_REPEATS):
    x12 = np.random.binomial(1, s_x, size=(N_x, 2))

    w_xy, w_xh, w_hy, w_hh, w_yy = {}, {}, {}, {}, {}
    for mode in stats.keys():
        w_xy[mode] = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
        w_xh[mode] = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
        w_hy[mode] = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
        w_hh[mode] = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
        w_yy[mode] = np.random.binomial(1, s_w_yy, size=(N_h, N_h))

    for iter_id in range(N_ITERS):
        h, y = {}, {}
        h['iWTA'], y['iWTA'] = iWTA(y0=w_xy['iWTA'] @ x12,
                                    h0=w_xh['iWTA'] @ x12,
                                    w_hy=w_hy['iWTA'])
        stats['nonzero'][repeat, iter_id] = np.count_nonzero(y['iWTA'],
                                                             axis=0).mean()

        for mode in ('kWTA', 'kWTA-fixed-k'):
            h[mode] = kWTA(w_xh[mode] @ x12, k=K_FIXED)
            # h[mode] = kWTA_different_k(w_xh[mode] @ x12, ks=np.count_nonzero(h['iWTA'], axis=0))
            y[mode] = w_xy[mode] @ x12 - w_hy[mode] @ h[mode]
        y['kWTA'] = kWTA_different_k(y['kWTA'],
                                     ks=np.count_nonzero(y['iWTA'], axis=0))
        y['kWTA-fixed-k'] = kWTA(y['kWTA-fixed-k'], k=K_FIXED)

        for mode in ('kWTA-fixed-k', 'kWTA', 'iWTA'):
            stats[mode][repeat, iter_id] = overlap2d(y[mode])
            update_weights(w_hy[mode], x_pre=h[mode], x_post=y[mode],
                           n_choose=NUM_TO_LEARN)

colormap = {
    'iWTA': 'green',
    'kWTA': 'blue',
    'kWTA-fixed-k': 'cyan'
}

fig, ax = plt.subplots()
ax.plot(range(N_ITERS), stats.pop('nonzero').mean(axis=0), lw=2, ls='dashed',
        label='nonzero', color='gray')

for key in stats.keys():
    mean = stats[key].mean(axis=0)
    std = stats[key].std(axis=0)
    ax.plot(range(N_ITERS), mean, lw=2, label=key, color=colormap[key])
    ax.fill_between(range(N_ITERS), mean + std, mean - std,
                    facecolor=colormap[key], alpha=0.3)
ax.set_title("Decorrelation")
ax.legend()
ax.set_xlabel('Iteration')
ax.set_ylabel('Overlap($y_1$, $y_2$)')
plt.savefig(RESULTS_DIR / "decorrelation.jpg")
plt.show()
