import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from kwta import kWTAi, update_weights, overlap, RESULTS_DIR, generate_k_active

N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy = 0.5, 0.1, 0.1, 0.1
N_repeat, N_epoch = 10, 100
K_FIXED = int(0.1 * N_y)
NUM_TO_LEARN = 5

stats = {
    mode: np.zeros((N_repeat, N_epoch), dtype=np.int32)
    for mode in ('overlap', 'nonzero_count')
}

for experiment in trange(N_repeat):
    # x = np.random.binomial(1, s_x, size=N_x)
    x = generate_k_active(n=N_x, k=int(s_x * N_x))

    w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))

    _, y0 = kWTAi(y0=w_xy @ x, h0=w_xh @ x, w_hy=w_hy)

    for iter_id in range(100):
        hi, yi = kWTAi(y0=w_xy @ x, h0=w_xh @ x, w_hy=w_hy)
        stats['overlap'][experiment, iter_id] = overlap(yi, y0)
        stats['nonzero_count'][experiment, iter_id] = yi.sum()
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
    ax.plot(range(N_epoch), mean, lw=2, label=key, color=colormap[key])
    ax.fill_between(range(N_epoch), mean + std, mean - std,
                    facecolor=colormap[key], alpha=0.3)
ax.set_title(f"Habituation. num_to_learn={NUM_TO_LEARN}")
ax.set_xlabel('iteration')
ax.set_ylabel('overlap(y, y0)')
ax.legend()
plt.savefig(RESULTS_DIR / "habituation.jpg")
# plt.show()
