"""
Connoisseur Data Sets Base.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import abc
import os
import tarfile
import zipfile
from urllib import request

import tensorflow as tf


class DataSet(metaclass=abc.ABCMeta):
    """Data Set base class."""
    SOURCE = None
    COMPACTED_FILE = None
    EXPECTED_SIZE = None

    def __init__(self, directory):
        self.directory = directory

    def download(self, override=False):
        os.makedirs(self.directory, exist_ok=True)
        file_name = os.path.join(self.directory, self.COMPACTED_FILE)

        if os.path.exists(file_name):
            stat = os.stat(file_name)
            if stat.st_size == self.EXPECTED_SIZE and not override:
                tf.logging.info('%s download skipped.', self.COMPACTED_FILE)
                return self

            tf.logging.info('copy corrupted. Re-downloading dataset.')

        file_name, _ = request.urlretrieve(self.SOURCE, file_name)
        stat = os.stat(file_name)
        tf.logging.info('%s downloaded (%i bytes).', self.COMPACTED_FILE,
                        stat.st_size)

        if stat.st_size != self.EXPECTED_SIZE:
            raise RuntimeError('File does not have expected size: (%i/%i)'
                               % (stat.st_size, self.EXPECTED_SIZE))
        return self

    def extract(self, override=False):
        zipped = os.path.join(self.directory, self.COMPACTED_FILE)
        unzipped = self.directory

        if os.path.isdir(unzipped) and not override:
            tf.logging.info('%s extraction skipped.', self.COMPACTED_FILE)
        else:
            extractor = self._get_specific_extractor(zipped)
            extractor.extractall(unzipped)
            extractor.close()

            tf.logging.info('paintings91 extracted.')

    @staticmethod
    def _get_specific_extractor(zipped):
        ext = os.path.splitext(zipped)[1]

        if ext in ('.tar', '.gz', '.tar.gz'):
            return tarfile.open(zipped)
        elif ext == '.zip':
            return zipfile.ZipFile(zipped, 'r')
        else:
            raise RuntimeError('Cannot extract %s. Unknown format.'
                               % zipped)

    def check(self):
        """Check dataset files."""
        raise NotImplementedError
