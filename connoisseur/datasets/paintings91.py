# coding: utf-8
"""Paintings91 Dataset.

Author: Lucas David -- <lucasolivdavid@gmail.com>
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
    EXTRACTED_FOLDER = 'Paintings91'

    def split_train_test(self):
        base_dir = self.full_data_path

        if os.path.exists(os.path.join(base_dir, 'train')):
            print('train-test splitting skipped.')
            return self

        images_file = os.path.join(base_dir, 'Labels', 'image_names.mat')
        labels_file = os.path.join(base_dir, 'Labels', 'labels.mat')
        train_file = os.path.join(base_dir, 'Labels', 'trainset.mat')
        test_file = os.path.join(base_dir, 'Labels', 'testset.mat')

        data = {}
        loadmat(train_file, mdict=data)
        loadmat(test_file, mdict=data)
        loadmat(images_file, mdict=data)
        loadmat(labels_file, mdict=data)

        # This stupid format and accents require us to call `replace`
        # to fix incorrect characters.
        data['image_names'] = np.array(
            list(map(lambda name: name[0][0].replace('Ã', 'É'),
                     data['image_names'])))
        X = data['image_names']
        y = data['labels']
        # Find absolute path.
        X = np.array([os.path.join(base_dir, 'Images', x) for x in X], copy=False)
        y = np.argmax(y, axis=1)

        train_indices, = np.where(data['trainset'].ravel())
        test_indices, = np.where(data['testset'].ravel())

        for phase, X, y in (('train', X[train_indices], y[train_indices]),
                            ('test', X[test_indices], y[test_indices])):
            for image, label in zip(X, y):
                painter_dir = os.path.join(base_dir, phase, str(label))
                os.makedirs(painter_dir, exist_ok=True)
                shutil.move(image, painter_dir)

        print('train-test splitting completed.')
        return self
