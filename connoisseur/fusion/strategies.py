import numpy as np

AVAILABLE_STRATEGIES = ['sum', 'mean', 'farthest', 'most_frequent',
                        'contrastive_mean']
__all__ = AVAILABLE_STRATEGIES


def sum(labels, distances, multi_class=True, t=0.0):
    if multi_class:
        return np.argmax(distances.sum(axis=-2), axis=-1)

    return (distances.sum(axis=-1) > t).astype(np.int)


def mean(labels, distances, multi_class=True, t=0.0):
    if multi_class:
        return np.argmax(distances.mean(axis=-2), axis=-1)

    return (distances.mean(axis=-1) > t).astype(np.int)


def contrastive_mean(labels, distances, multi_class=True, t=0.0):
    return (distances.mean(axis=-1) <= t).astype(np.int)


def farthest(labels, distances, multi_class=True, t=0.0):
    if multi_class:
        return np.argmax(distances.max(axis=-2), axis=-1)

    n_samples = distances.shape[0]
    patches = np.argmax(np.absolute(distances), axis=-1)
    return (distances[range(n_samples), patches] > 0).astype(np.int)


def most_frequent(labels, distances, multi_class=True, t=0.0):
    return np.array([np.argmax(np.bincount(patch_labels))
                     for patch_labels in labels], copy=False)
