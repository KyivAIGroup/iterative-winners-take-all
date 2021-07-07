import numpy as np
from scipy.spatial.distance import pdist, squareform


def overlap(x1, x2):
    return (x1 & x2).sum()


def cosine_similarity(x1, x2):
    return x1.dot(x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def generate_k_active(n, k):
    x = np.zeros(n, dtype=np.int32)
    active = np.random.choice(n, size=k, replace=False)
    x[active] = 1
    return x


def compute_discriminative_factor(array, labels):
    # array shape is (n_samples, n_features)
    dist = pdist(array)
    dist = squareform(dist)
    factor = []
    for label in np.unique(labels):
        mask_same = labels == label
        if mask_same.sum() == 1:
            # empty dist_same
            continue
        dist_same = dist[mask_same][:, mask_same]
        dist_other = dist[mask_same][:, ~mask_same]
        # squareform excludes zeros on the main diagonal of dist_same
        dist_same = squareform(dist_same).mean()
        if dist_same != 0:
            # all vectors degenerated in a single vector
            factor.append(dist_other.mean() / dist_same)
    if len(factor) == 0:
        return None
    factor = np.mean(factor)
    return factor
