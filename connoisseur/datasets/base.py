"""
Connoisseur Data Sets Base Class.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import abc
import os
import tarfile
import zipfile
import logging
from urllib import request

import tensorflow as tf

from ..utils import image as img_utils

logger = logging.getLogger('connoisseur')


class DataSet(metaclass=abc.ABCMeta):
    """Data Set.

    Holds data pointers for a data set, such as its features' values and/or
    supervised tensor. It can download, extract, pre-process the data and
    serve batches.

    The method `load` should be overridden as it depends on the format and
    nature of the data set (e.g.: images are not loaded the same as text).
    """

    DEFAULT_PARAMETERS = {}
    NAME = None

    def __init__(self, **parameters):
        for k, v in self.DEFAULT_PARAMETERS.items():
            if k not in parameters:
                parameters[k] = v

        self.parameters = parameters

        self.train_data, self.train_target = None, None
        self.validation_data, self.validation_target = None, None
        self.test_data, self.test_target = None, None
        self._percentage_transferred = None

    @property
    def loaded(self):
        """Indicates if data set was loaded or not.

        :return: bool, True if data indicates a loaded tensor.
        """
        return self.train_data is not None

    def download(self, override=False):
        """Download Data Set from the address indicated by `source` parameter.

        :param override: ignore old files and always download.
        :return: self
        """
        print('Downloading:', end=' ', flush=True)
        p = self.parameters

        file_names = p['file_name']
        if not isinstance(file_names, (list, tuple)):
            file_names = [file_names]
        file_names = [os.path.join(p['save_in'], name) for name in file_names]

        sources = p['source']
        if not isinstance(sources, (list, tuple)):
            sources = [sources]

        expected_sizes = p['expected_size']
        if not isinstance(expected_sizes, (list, tuple)):
            expected_sizes = [expected_sizes]

        os.makedirs(p['save_in'], exist_ok=True)

        for file_name, source, expected_size in zip(file_names, sources,
                                                    expected_sizes):
            if os.path.exists(file_name) and not override:
                stat = os.stat(file_name)
                print('(skipped)')
            else:
                file_name, _ = request.urlretrieve(
                    source, file_name,
                    reporthook=self._download_progress_hook)
                stat = os.stat(file_name)
                print('\nDone. %i bytes transferred.' % stat.st_size)

            if stat.st_size != expected_size:
                raise RuntimeError('File does not have expected size: (%i/%i)'
                                   % (stat.st_size, p['expected_size']))
        return self

    def extract(self, override=False):
        """Extract Downloaded Data Set.

        A folder `name` will be created in `save_in` directory.

        :param override: ignore files and extract regardless.
        :return: self
        """
        print('Extracting...', end=' ', flush=True)

        p = self.parameters

        file_names = p['file_name']
        if not isinstance(file_names, (list, tuple)):
            file_names = [file_names]

        for file_name in file_names:
            zipped = os.path.join(p['save_in'], file_name)
            unzipped = os.path.join(p['save_in'])

            if os.path.isdir(unzipped) and not override:
                print('(skipped)')
            else:
                extractor = self._get_specific_extractor(zipped)
                extractor.extractall(unzipped)
                extractor.close()

                print('Done. Placed at %s.' % unzipped)

        return self

    def load(self):
        """Load Data as a Tensorflow's `Tensor`.

        :return: self
        """
        raise NotImplementedError

    def preprocess(self):
        """Submit Data to Pre-processing.

        For example, one can perform scaling or data whitening. By default,
        does nothing.

        :return: self
        """
        return self

    def as_batches(self, phase='train', size=None):
        """Returns The Next Data Batch.

        :param phase: str, default='train'
            Current phase taking place. Options are: train, validation, test.
        :param size: int, default=None
            Batch size. If None, will use default passed in the construction of
            this data set.

        :return: tuple (data, target)
        """
        assert phase in ('train', 'validation', 'test')
        assert self.loaded, ('Cannot ask for next batch in '
                             'data sets which are not loaded.')

        size = size or self.parameters['batch_size']
        data = getattr(self, '%s_data' % phase)
        target = getattr(self, '%s_target' % phase)

        return tf.train.batch([data, target], batch_size=size)

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

    def _download_progress_hook(self, count, block_size, total_size):
        """A hook to report the progress of a download.

        This is mostly intended for users with slow internet connections.
        Reports every 1% change in download progress.

        """
        percent = int(count * block_size * 100 / total_size)

        if self._percentage_transferred != percent:
            if percent % 25 == 0:
                print('%i%%' % percent, end='', flush=True)
            elif percent % 5:
                print('.', end='', flush=True)

            self._percentage_transferred = percent


class ImageDataSet(DataSet, metaclass=abc.ABCMeta):
    """Image Data Set.

    Base class used by data sets of images.
    """

    def __init__(self, **parameters):
        super().__init__(**parameters)

        self.train_image_names = None
        self.validation_image_names = None
        self.test_image_names = None

    def preprocess(self):
        """Pre-process image samples.

        Pre-processing consists on cropping and padding images to make sure
        they are on the same size, as well as performing pixel whitening.

        :return: self
        """
        params = self.parameters

        for phase in ('train', 'validation', 'test'):
            # Crop and pad image to get them to all expected lengths.
            data = getattr(self, '%s_data' % phase)

            if data is None:
                # Skip phase if not set.
                continue

            image = img_utils.resize_image_with_crop_or_pad(data,
                                                            params['height'],
                                                            params['width'])
            image.set_shape([params['height'], params['width'], params['channels']])
            data = tf.image.per_image_whitening(image)

            setattr(self, '%s_data' % phase, data)

        return self
