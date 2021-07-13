import numpy as np


def compute_loss(output, labels):
    assert len(output) == len(labels)
    output = output / np.linalg.norm(output, axis=1, keepdims=True)
    cosine_similarity = output.dot(output.T)
    loss = []
    labels_unique = np.unique(labels)
    clustering = len(labels) > len(labels_unique)
    for label in labels_unique:
        mask_same = labels == label
        cos_other = cosine_similarity[mask_same][:, ~mask_same]
        cos_other = cos_other.mean()
        if clustering:
            cos_same = cosine_similarity[mask_same][:, mask_same]
            n = cos_same.shape[0]
            if n == 1:
                continue
            ii, jj = np.triu_indices(n, k=1)
            cos_same = cos_same[ii, jj].mean()
            loss.append(1 - cos_same + cos_other)
        else:
            loss.append(cos_other)
    if len(loss) == 0:
        return None
    loss = np.mean(loss)
    return loss


def compute_accuracy(output, labels):
    assert len(output) == len(labels)
    output = output / np.linalg.norm(output, axis=1, keepdims=True)
    centroids = [output[labels == l].mean(axis=0) for l in np.unique(labels)]
    centroids = np.vstack(centroids)
    cosine_similarity = output.dot(centroids.T)
    labels_pred = cosine_similarity.argmax(axis=1)
    accuracy = (labels == labels_pred).mean()
    return accuracy
