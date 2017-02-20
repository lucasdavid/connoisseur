import numpy as np

from keras.utils.generic_utils import get_from_module

AVAILABLE_STRATEGIES = ['sum', 'farthest', 'most_frequent']
__all__ = AVAILABLE_STRATEGIES + ['get']


def sum(labels, distances, multi_class=True):
    if multi_class:
        return np.argmax(distances.sum(axis=-2), axis=-1)

    return (distances.sum(axis=-1) > 0).astype(np.int)


def contrastive_avg(labels, distances):
    return distances.mean(axis=-1)


def farthest(labels, distances, multi_class=True):
    if multi_class:
        return np.argmax(distances.max(axis=-2), axis=-1)

    n_samples = distances.shape[0]
    patches = np.argmax(np.absolute(distances), axis=-1)
    return (distances[range(n_samples), patches] > 0).astype(np.int)


def most_frequent(labels, distances, multi_class=True):
    return np.array([np.argmax(np.bincount(patch_labels))
                     for patch_labels in labels], copy=False)


def get(strategy):
    if strategy is None:
        return sum
    return get_from_module(strategy, globals(), 'fusion strategy')
