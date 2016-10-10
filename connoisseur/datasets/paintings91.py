import os
import shutil

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat

from .base import DataSet


class Paintings91(DataSet):
    """Painting91 Data set."""

    SOURCE = 'http://cat.cvc.uab.es/~joost/data/Paintings91.zip'
    COMPACTED_FILE = 'paintings91.zip'
    EXPECTED_SIZE = 681584272

    def _load_images_and_labels(self):
        images_file = os.path.join(self.directory, 'Labels', 'image_names.mat')
        labels_file = os.path.join(self.directory, 'Labels', 'labels.mat')

        # Data is represented at a column vector. Coverts it to a list.
        X = loadmat(images_file)['image_names']
        # This stupid format and accents require us to call `replace` to fix
        # incorrect characters.
        X = np.array(list(map(lambda name: name[0][0].replace('Ã', 'É'), X)))
        y = np.argmax(loadmat(labels_file)['labels'], axis=1)

        return X, y

    def extract(self, override=False):
        super().extract(override=override)

        dataset_dir = os.path.join(self.directory,
                                   os.path.splitext(self.COMPACTED_FILE)[0])
        unzipped = os.path.join(dataset_dir, 'Paintings91')

        if os.path.exists(unzipped):
            # this unzipping will end up being paintings91/Paintings91.
            # Move all files to an upper tree level.
            for p in os.listdir(unzipped):
                os.rename(os.path.join(unzipped, p),
                          os.path.join(self.directory, p))

            # Remove leaf folder, indicating this operation was already done.
            os.rmdir(unzipped)

        return self

    def check(self):
        if not os.path.exists(self.directory):
            raise RuntimeError('Data set not found. Have you downloaded '
                               'and extracted it first?')

        X, y = self._load_images_and_labels()

        images_folder = os.path.join(self.directory, 'Images')
        X_real = []

        for i in range(91):
            # Translates dataset to directory format expected by Keras.
            paintings = X[y == i].tolist()
            artist = '_'.join(paintings[0].split('_')[:-1])
            artist_folder = os.path.join(images_folder, artist)

            os.makedirs(artist_folder, exist_ok=True)

            for painting in paintings:
                try:
                    shutil.move(os.path.join(images_folder, painting),
                                artist_folder)
                except shutil.Error:
                    # File was already copied.
                    pass
                X_real.append(os.path.join(artist_folder, painting))

        X = np.array(X_real)

        for painting in X:
            assert os.path.exists(painting), \
                'a painting is missing: %s' % painting

        return self

    def as_keras_generator(self):
        return ImageDataGenerator(
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            zca_whitening=False,
            width_shift_range=.2,
            height_shift_range=.2,
            rescale=1. / 255,
            fill_mode='wrap')
