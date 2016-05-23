import abc


class Connoisseur(metaclass=abc.ABCMeta):
    """Connoisseur Base Class."""

    def __init__(self, learning_rate=0.001, dropout=.5, network=None, optimizer=None):
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.network = network
        self.optimizer = None

    def fit(self, data_set):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
