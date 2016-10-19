from keras.engine import Model
from keras.utils.generic_utils import get_from_module
from sklearn.base import BaseEstimator

from . import strategies
from .base import KerasFusion, SkLearnFusion


def get(fusion_system_or_model):
    if fusion_system_or_model is None:
        return KerasFusion

    # If `fusion_system_or_model` is a model, identify which kind and returns
    # the appropriate fusion system class.
    if isinstance(fusion_system_or_model, Model):
        return KerasFusion
    if isinstance(fusion_system_or_model, BaseEstimator):
        return SkLearnFusion

    return get_from_module(fusion_system_or_model, globals(), 'fusion system')


__all__ = ['KerasFusion', 'SkLearnFusion', 'strategies']
