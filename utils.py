import numpy as np
from scipy.spatial.distance import squareform


def overlap(x1, x2):
    return (x1 & x2).sum()


def generate_k_active(n, k):
    x = np.zeros(n, dtype=np.int32)
    active = np.random.choice(n, size=k, replace=False)
    x[active] = 1
    return x


def compute_loss(output, labels):
    assert len(output) == len(labels)
    output = output / np.linalg.norm(output, axis=1, keepdims=True)
    cos = output.dot(output.T)
    loss = []
    for label in np.unique(labels):
        mask_same = labels == label
        cos_same = cos[mask_same][:, mask_same]
        np.fill_diagonal(cos_same, val=0)
        if cos_same.size == 1:
            continue
        cos_other = cos[mask_same][:, ~mask_same]
        # squareform excludes zeros on the main diagonal of cos_same
        cos_same = squareform(cos_same).mean()
        if cos_same != 0:
            # all vectors degenerated in a single vector
            loss.append(1 - cos_same + cos_other.mean())
    if len(loss) == 0:
        return None
    loss = np.mean(loss)
    return loss
