"""VanGogh Dataset.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import os

from keras.preprocessing.image import ImageDataGenerator

from .base import DataSet


class VanGogh(DataSet):
    SOURCE = 'https://ndownloader.figshare.com/files/5870145'
    COMPACTED_FILE = 'vangogh.zip'
    EXPECTED_SIZE = 5707509034

    def check(self):
        base_path = os.path.join(self.directory, 'vgdb_2016')
        assert os.path.exists(base_path), ('Data set not found. Have '
                                           'you downloaded and extracted '
                                           'it first?')
        return self

    def as_keras_generator(self):
        return ImageDataGenerator(
            width_shift_range=.9,
            height_shift_range=.9,
            rescale=1. / 255,
            fill_mode='wrap')
