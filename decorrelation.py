import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from kwta import kWTA, kWTAi, update_weights, overlap

N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy = 0.5, 0.1, 0.1, 0.1
N_repeat, N_epoch = 10, 100
K_FIXED = int(0.1 * N_y)
NUM_TO_LEARN = 5

stats = {mode: np.zeros((N_repeat, N_epoch), dtype=np.float32)
         for mode in ('kwta-fixed-k', 'kwta', 'iwta')}

for experiment in trange(N_repeat):
    x1 = np.random.binomial(1, s_x, size=N_x)
    x2 = np.random.binomial(1, s_x, size=N_x)

    w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))

    for iter_id in range(100):
        h1, y1 = kWTAi(y0=w_xy @ x1, h0=w_xh @ x1, w_hy=w_hy)
        h2, y2 = kWTAi(y0=w_xy @ x2, h0=w_xh @ x1, w_hy=w_hy)
        stats['iwta'][experiment, iter_id] = overlap(y1, y2)

        y1_kwta = w_xy @ x1 - w_hy @ (w_xh @ x1)
        y2_kwta = w_xy @ x2 - w_hy @ (w_xh @ x2)
        overlap_kwta_fixed_k = overlap(kWTA(y1_kwta, k=K_FIXED),
                                       kWTA(y2_kwta, k=K_FIXED))
        overlap_kwta = overlap(kWTA(y1_kwta, k=np.count_nonzero(y1)),
                               kWTA(y2_kwta, k=np.count_nonzero(y2)))
        stats['kwta-fixed-k'][experiment, iter_id] = overlap_kwta_fixed_k
        stats['kwta'][experiment, iter_id] = overlap_kwta

        update_weights(w_hy, x_pre=h1, x_post=y1, n_choose=NUM_TO_LEARN)
        # update_weights(w_xy, x_pre=x1, x_post=y1, n_choose=1)

colormap = {
    'iwta': 'green',
    'kwta': 'blue',
    'kwta-fixed-k': 'cyan'
}

fig, ax = plt.subplots()

for key in stats.keys():
    mean = stats[key].mean(axis=0)
    std = stats[key].std(axis=0)
    ax.plot(range(N_epoch), mean, lw=2, label=key, color=colormap[key])
    ax.fill_between(range(N_epoch), mean + std, mean - std,
                    facecolor=colormap[key], alpha=0.3)
ax.set_title(f"Decorrelation. num_to_learn={NUM_TO_LEARN}")
ax.legend()
ax.set_xlabel('iteration')
ax.set_ylabel('overlap(y1, y2)')
plt.show()
