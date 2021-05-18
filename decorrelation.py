import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from kwta import kWTA, iWTA, update_weights, overlap, RESULTS_DIR

N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh = 0.5, 0.1, 0.1, 0.1, 0.05
N_REPEATS, N_ITERS = 10, 100
K_FIXED = int(0.15 * N_y)
NUM_TO_LEARN = 5

stats = {mode: np.zeros((N_REPEATS, N_ITERS), dtype=np.float32)
         for mode in ('kWTA-fixed-k', 'kWTA', 'iWTA')}
n_active = np.zeros((N_REPEATS, N_ITERS), dtype=np.float32)

for repeat in trange(N_REPEATS):
    x1 = np.random.binomial(1, s_x, size=N_x)
    x2 = np.random.binomial(1, s_x, size=N_x)

    w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
    w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))

    for iter_id in range(100):
        h1, y1 = iWTA(y0=w_xy @ x1, h0=w_xh @ x1, w_hy=w_hy, w_hh=w_hh)
        h2, y2 = iWTA(y0=w_xy @ x2, h0=w_xh @ x1, w_hy=w_hy, w_hh=w_hh)
        stats['iWTA'][repeat, iter_id] = overlap(y1, y2)

        y1_kwta_pre = w_xy @ x1 - w_hy @ (w_xh @ x1)
        y2_kwta_pre = w_xy @ x2 - w_hy @ (w_xh @ x2)
        overlap_kwta_fixed_k = overlap(kWTA(y1_kwta_pre, k=K_FIXED),
                                       kWTA(y2_kwta_pre, k=K_FIXED))
        overlap_kwta = overlap(kWTA(y1_kwta_pre, k=np.count_nonzero(y1)),
                               kWTA(y2_kwta_pre, k=np.count_nonzero(y2)))
        stats['kWTA-fixed-k'][repeat, iter_id] = overlap_kwta_fixed_k
        stats['kWTA'][repeat, iter_id] = overlap_kwta
        n_active[repeat, iter_id] = np.count_nonzero(y1)

        update_weights(w_hy, x_pre=h1, x_post=y1, n_choose=NUM_TO_LEARN)
        # update_weights(w_xy, x_pre=x1, x_post=y1, n_choose=1)

colormap = {
    'iWTA': 'green',
    'kWTA': 'blue',
    'kWTA-fixed-k': 'cyan'
}

fig, ax = plt.subplots()

n_active = n_active.mean(axis=0)
print(f"num. of active neurons in y1: {n_active}")
for key in stats.keys():
    mean = stats[key].mean(axis=0)
    std = stats[key].std(axis=0)
    ax.plot(range(N_ITERS), mean, lw=2, label=key, color=colormap[key])
    ax.fill_between(range(N_ITERS), mean + std, mean - std,
                    facecolor=colormap[key], alpha=0.3)
ax.set_title(f"Decorrelation. num_to_learn={NUM_TO_LEARN}")
ax.legend()
ax.set_xlabel('iteration')
ax.set_ylabel('overlap(y1, y2)')
plt.savefig(RESULTS_DIR / "decorrelation.jpg")
# plt.show()
