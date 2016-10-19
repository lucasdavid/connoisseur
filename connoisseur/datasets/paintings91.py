# coding: utf-8
"""Paintings91 Dataset.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import os
import shutil

import numpy as np
from scipy.io import loadmat

from .base import DataSet


class Paintings91(DataSet):
    """Painting91 Data set."""

    SOURCE = 'http://cat.cvc.uab.es/~joost/data/Paintings91.zip'
    EXPECTED_SIZE = 681584272
    DATA_SUB_DIR = 'Paintings91/Images'

    def _load_images_and_labels(self):
        base_dir = self.base_dir
        images_file = os.path.join(base_dir, 'Labels', 'image_names.mat')
        labels_file = os.path.join(base_dir, 'Labels', 'labels.mat')

        # Data is represented at a column vector. Coverts it to a list.
        X = loadmat(images_file)['image_names']
        # This stupid format and accents require us to call `replace` to fix
        # incorrect characters.
        X = np.array(list(map(lambda name: name[0][0].replace('Ã', 'É'), X)))
        y = np.argmax(loadmat(labels_file)['labels'], axis=1)

        return X, y

    def check(self):
        data_path = self.full_data_path
        assert os.path.exists(data_path), 'Data set not found. Have you downloaded and extracted it first?'
        X, y = self._load_images_and_labels()

        images_folder = os.path.join(data_path,
                                     'Paintings91', 'Images')
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
            assert os.path.exists(painting), 'a painting is missing: %s' % painting
        return self
