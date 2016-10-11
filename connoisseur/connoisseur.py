"""Connoisseur Base.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import abc


class Connoisseur(metaclass=abc.ABCMeta):
    """Connoisseur Base Class.

    Attributes:
        model_ :
    """

    def __init__(self, constants=None, build=False):
        self.constants = constants

        self.model_ = self.dataset_ = None

        if build:
            self.build_model()
            self.build_dataset()

    @abc.abstractmethod
    def build_model(self):
        """Build the machine learning model used by this connoisseur."""

    @abc.abstractmethod
    def build_dataset(self):
        """Build the dataset studied by this connoisseur."""

    @abc.abstractmethod
    def data(self, phase='train'):
        """Get a data iterator for a specific phase.

        Args:
            phase: possible values are: ('train', 'test').
                   The phase in which the connoisseur currently is.

        Returns:
            A data iterator that returns the tuple (X, y) for the paintings.

        """

    def __del__(self):
        if self.model_:
            del self.model_
        if self.dataset_:
            del self.dataset_
