"""Connoisseur Base.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import abc


class Connoisseur(metaclass=abc.ABCMeta):
    """Connoisseur Base Class."""

    def build(self):
        raise NotImplementedError

    def data(self, phase='training'):
        raise NotImplementedError
