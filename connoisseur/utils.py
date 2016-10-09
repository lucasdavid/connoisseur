"""Connoisseur Utils.


Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import json
import time


class Timer:
    """Pretty time counter."""

    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def pretty_elapsed(self):
        m, s = divmod(self.elapsed(), 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str

    def __str__(self):
        return self.pretty_elapsed()


class Constants:
    """Loads info from a .json file.

    Useful for when executing from multiple different environments.

    """

    def __init__(self, filename='./constants.json', raise_on_not_found=False):
        self.filename = filename
        self.raise_on_not_found = raise_on_not_found

        self._data = self.logging_dir = self.data_dir = self.batch_size = \
            self.image_shape = self.classes = self.seed = self.n_iterations = \
            self.n_samples_per_epoch = self.device = None

    def load(self):
        try:
            with open(self.filename) as f:
                data = json.load(f)
        except IOError:
            if self.raise_on_not_found: raise
            data = {}

        self._data = data

        self.n_iterations = data.get('n_iterations', 100)
        self.n_samples_per_epoch = data.get('n_samples_per_epoch', 1024)
        self.data_dir = data.get('data_dir', '.')
        self.logging_dir = data.get('logging_dir', '.')
        self.batch_size = data.get('batch_size', 256)
        self.image_shape = data.get('image_shape', (224, 224, 3))
        self.classes = data.get('classes', None)
        self.device = data.get('device', '/gpu:0')
        self.seed = data.get('seed', None)

        return self
