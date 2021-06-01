import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from kwta import kWTA, iWTA, kWTA_different_k
from constants import RESULTS_DIR
from utils import generate_k_active

N_x, N_y, N_h = 100, 200, 200
s_x, s_w_xy, s_w_xh, s_w_hy, s_w_hh = 0.1, 0.1, 0.1, 0.1, 0.05
N_REPEAT, N_SPLIT = 100, 10
K_FIXED = int(0.1 * N_y)

def generate_similar_input(x, n_split=N_SPLIT):
    # Generates n_split vectors with the same num. of active units.
    # The similarity between the output tensor and input vector increases
    #   with each column (from 0 to 1).
    # 'x' is a (N,) vec
    idx_pool = x.nonzero()[0]
    no_overlap_idx = (x == 0).nonzero()[0]
    k = len(idx_pool)
    n_idx_take = np.linspace(0, k, num=n_split, dtype=int)
    x_similar = np.zeros((x.shape[0], n_split), dtype=np.int32)
    for i, k_common in enumerate(n_idx_take):
        active = np.random.choice(idx_pool, size=k_common, replace=False)
        active_no_overlap = np.random.choice(no_overlap_idx, size=k - k_common,
                                             replace=False)
        active = np.append(active, active_no_overlap)
        x_similar[active, i] = 1
    assert (x_similar[:, -1] == x).all()
    assert (x_similar.sum(axis=0) == k).all()
    return x_similar  # (N, n_split)


def cosine_similarity(x_tensor):
    x_orig = x_tensor[:, -1]  # the last column
    norm = np.linalg.norm(x_tensor, axis=0)  # must be the same
    similarity = np.divide(x_orig.dot(x_tensor),
                           np.linalg.norm(x_orig) * norm,
                           out=np.zeros(norm.shape),
                           where=norm != 0)
    return similarity


stats = {mode: np.zeros((N_REPEAT, N_SPLIT), dtype=np.float32)
         for mode in ('kWTA-fixed-k', 'kWTA', 'iWTA')}
n_active = np.zeros((N_REPEAT, N_SPLIT), dtype=np.float32)

similarity_x = None

for repeat in trange(N_REPEAT):
    x = generate_k_active(n=N_x, k=10)
    x_similar = generate_similar_input(x)

    w_xy = np.random.binomial(1, s_w_xy, size=(N_y, N_x))
    w_xh = np.random.binomial(1, s_w_xh, size=(N_h, N_x))
    w_hy = np.random.binomial(1, s_w_hy, size=(N_y, N_h))
    w_hh = np.random.binomial(1, s_w_hh, size=(N_h, N_h))

    y_kwta_pre = w_xy @ x_similar - w_hy @ (w_xh @ x_similar)
    _, y_similar = iWTA(y0=w_xy @ x_similar, h0=w_xh @ x_similar, w_hy=w_hy, w_hh=w_hh)
    y_kwta_fixed_k = kWTA(y_kwta_pre, k=K_FIXED)
    n_active_batch = np.count_nonzero(y_similar, axis=0)
    y_kwta = kWTA_different_k(y_kwta_pre, ks=n_active_batch)

    similarity_x = cosine_similarity(x_similar)  # same for all repeats
    stats['iWTA'][repeat] = cosine_similarity(y_similar)
    stats['kWTA'][repeat] = cosine_similarity(y_kwta)
    stats['kWTA-fixed-k'][repeat] = cosine_similarity(y_kwta_fixed_k)
    n_active[repeat] = n_active_batch

colormap = {
    'iWTA': 'green',
    'kWTA': 'blue',
    'kWTA-fixed-k': 'cyan'
}

fig, ax = plt.subplots()
ax.set_aspect(1)

n_active = n_active.mean(axis=0)
print(f"{n_active=}")
for key in stats.keys():
    mean = stats[key].mean(axis=0)
    std = stats[key].std(axis=0)
    ax.plot(similarity_x, mean, lw=2, label=key, color=colormap[key])
    ax.fill_between(similarity_x, mean + std, mean - std,
                    facecolor=colormap[key], alpha=0.3)
ax.set_title("Similarity preservation")
ax.legend()
ax.set_xlabel('x similarity')
ax.set_ylabel('y similarity')
plt.savefig(RESULTS_DIR / "similarity_preservation.jpg")
# plt.show()
