"""
Metrics, used in the paper.
"""
import numpy as np


def compute_error(output, labels):
    """
    Compute the error function.

    Parameters
    ----------
    output : (S, N) np.ndarray
        A sample-by-neurons transposed output activations.
    labels : (S,) np.ndarray
        Sample class labels (ids).

    Returns
    -------
    error : float
        The total error.
    """
    assert len(output) == len(labels)
    norm = np.linalg.norm(output, axis=1, keepdims=True)
    norm += 1e-10  # add a small value to avoid division by zero
    output = output / norm
    cosine_similarity = output.dot(output.T)
    error = []
    labels_unique = np.unique(labels)
    # Labels repeat in clustering experiment and are unique in decorrelation
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
            error.append(1 - cos_same + cos_other)
        else:
            error.append(cos_other)
    if len(error) == 0:
        return None
    error = np.mean(error)
    return error


def cluster_centroids(output, labels):
    centroids = [output[labels == l].mean(axis=0) for l in np.unique(labels)]
    centroids = np.vstack(centroids)
    return centroids


def compute_accuracy(output, labels):
    """
    Compute the accuracy by checking the predicted labels with the true
    `labels`. The predicted label `l` of a sample :math:`x` is computed as

    .. math::
        l = \argmin_i \cos(x, x_c^i)

    where :math:`\cos` is the cosine similarity between two vectors and
    :math:`x_c^i` is the mean output vector (centroid) for the class `i`.

    Parameters
    ----------
    output : (S, N) np.ndarray
        A sample-by-neurons transposed output activations.
    labels : (S,) np.ndarray
        Sample class labels (ids).

    Returns
    -------
    accuracy : float
        The accuracy.
    """
    assert len(output) == len(labels)
    norm = np.linalg.norm(output, axis=1, keepdims=True)
    norm += 1e-10  # add a small value to avoid division by zero
    output = output / norm
    centroids = cluster_centroids(output, labels)
    cosine_similarity = output.dot(centroids.T)
    labels_pred = cosine_similarity.argmax(axis=1)
    accuracy = (labels == labels_pred).mean()
    return accuracy


def compute_convergence(output, output_prev):
    """
    Compute the convergence by comparing with the output from the previous
    iteration.

    The convergence is measured as the mean of a XOR operation on two vectors.

    Parameters
    ----------
    output, output_prev : np.ndarray
        Current and previous iteration binary outputs.

    Returns
    -------
    float
        The model convergence between 0 (fully converged) and 1 (fully chaotic)
    """
    if output is None or output_prev is None:
        return None
    return (output ^ output_prev).mean()
