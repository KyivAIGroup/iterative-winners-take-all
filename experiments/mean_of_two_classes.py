import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import trange

from kwta import iWTA, update_weights, RESULTS_DIR, kWTA, kWTA_different_k, overlap, cosine_similarity

N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh, s_w_yy = 0.5, 0.1, 0.1, 0.1, 0.1, 0.02
N_REPEATS, N_ITERS = 10, 100
K_FIXED = int(0.1 * N_y)


stats = {
    mode: np.zeros((N_REPEATS, N_ITERS), dtype=np.float32)
    for mode in ('iWTA', 'kWTA', 'kWTA-fixed-k')
}
n_active = np.zeros((N_REPEATS, N_ITERS), dtype=np.float32)


for repeat in range(N_REPEATS):
    x12 = np.random.binomial(1, s_x, size=(2, N_x))
    x1, x2 = x12
    x1_noisy, x2_noisy = x12 + np.random.binomial(1, 0.05, size=x12.shape)

    w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
    w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))
    w_yy = np.random.binomial(1, s_w_yy, size=(N_y, N_y))

    for x in x12:
        y0 = w_xy @ x
        h0 = w_xh @ x
        for iteration in range(N_ITERS):
            h, y = iWTA(y0=y0, h0=h0, w_hy=w_hy, w_yy=w_yy)
            update_weights(w_hy, x_pre=x, x_post=h, n_choose=2)
            update_weights(w_xy, x_pre=x, x_post=y, n_choose=2)

    h1, y1 = iWTA(y0=w_xy @ x1, h0=w_xh @ x1, w_hy=w_hy, w_yy=w_yy)
    h2, y2 = iWTA(y0=w_xy @ x2, h0=w_xy @ x2, w_hy=w_hy, w_yy=w_yy)
    z = y1 & y2

    h1_noisy, y1_noisy = iWTA(y0=w_xy @ x1_noisy, h0=w_xh @ x1_noisy, w_hy=w_hy, w_yy=w_yy)

    print(f"{overlap(y1_noisy, y1) - overlap(y1_noisy, z)=}, {overlap(y1_noisy, y2) - overlap(y1_noisy, z)=}, {overlap(y1_noisy, z)=}")
