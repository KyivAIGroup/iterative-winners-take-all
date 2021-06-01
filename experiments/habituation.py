import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from kwta import iWTA, update_weights
from constants import RESULTS_DIR
from utils import overlap, generate_k_active

N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh = 0.5, 0.1, 0.1, 0.1, 0.1
N_REPEATS, N_ITERS = 10, 100
K_FIXED = int(0.1 * N_y)
NUM_TO_LEARN = 5

stats = {
    mode: np.zeros((N_REPEATS, N_ITERS), dtype=np.int32)
    for mode in ('overlap', 'nonzero_count')
}

for repeat in trange(N_REPEATS):
    # x = np.random.binomial(1, s_x, size=N_x)
    x = generate_k_active(n=N_x, k=int(s_x * N_x))

    w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
    w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))

    _, y0 = iWTA(y0=w_xy @ x, h0=w_xh @ x, w_hy=w_hy, w_hh=w_hh)

    for iter_id in range(100):
        hi, yi = iWTA(y0=w_xy @ x, h0=w_xh @ x, w_hy=w_hy, w_hh=w_hh)
        stats['overlap'][repeat, iter_id] = overlap(yi, y0)
        stats['nonzero_count'][repeat, iter_id] = yi.sum()
        update_weights(w_hy, x_pre=hi, x_post=yi, n_choose=NUM_TO_LEARN)
        # update_weights(w_xy, x_pre=x, x_post=yi, n_choose=10)

colormap = {
    'overlap': 'blue',
    'nonzero_count': 'cyan'
}

fig, ax = plt.subplots()

for key in stats.keys():
    mean = stats[key].mean(axis=0)
    std = stats[key].std(axis=0)
    ax.plot(range(N_ITERS), mean, lw=2, label=key, color=colormap[key])
    ax.fill_between(range(N_ITERS), mean + std, mean - std,
                    facecolor=colormap[key], alpha=0.3)
ax.set_title(f"Habituation. num_to_learn={NUM_TO_LEARN}")
ax.set_xlabel('iteration')
ax.set_ylabel('overlap(y, y0)')
ax.legend()
plt.savefig(RESULTS_DIR / "habituation.jpg")
# plt.show()
