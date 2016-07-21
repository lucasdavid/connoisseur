import os

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

from . import base
from ..utils import transformations


class Paintings91(base.ImageDataSet):
    DEFAULT_PARAMETERS = {
        'file_name': 'Paintings91.zip',
        'source': 'http://cat.cvc.uab.es/~joost/data/Paintings91.zip',
        'expected_size': 681584272,

        'batch_size': 50,
        'height': 300,
        'width': 300,
        'channels': 3,

        'train_validation_test_split': [.8, .2],

        'n_classes': 91,
        'save_in': '/tmp',
        'n_epochs': None,
    }

    def __init__(self, **parameters):
        super().__init__('Paintings91', **parameters)

    def load(self, override=False):
        self.download(override=override).extract(override=override)

        params = self.parameters

        base_folder = os.path.join(params['save_in'], self.name, 'Paintings91')

        if not os.path.exists(base_folder):
            raise RuntimeError('Data set not found. Have you downloaded '
                               'and extracted it first?')

        images_folder = os.path.join(base_folder, 'Images')
        labels_folder = os.path.join(base_folder, 'Labels', 'labels.mat')

        # Data is represented at a column vector. Coverts it to a list.
        image_names = \
            loadmat(os.path.join(base_folder, 'Labels', 'image_names.mat'))[
                'image_names']
        # Stupid format and accents require us to call
        # `replace` to fix incorrect characters.
        image_names = np.array(
            [os.path.join(images_folder, file_name[0][0].replace('Ã', 'É'))
             for file_name in image_names])

        for image in image_names:
            # Let's just make sure all files are in place
            # before any hard computation.
            assert os.path.exists(image)

        # Be aware: labels are one-hot encoded already.
        target = loadmat(labels_folder)['labels']

        # Let's check if problem is ill-sampled by defining a threshold of 6
        # samples and asserting that standard deviation of the occurrence of
        # a class is greater 6 samples, where the average occurrence of a class
        # is approximately 46.
        target_decoded = transformations.OneHot.decode(target)
        classes, classes_count = np.unique(target_decoded, return_counts=True)
        assert classes_count.std() <= 6
        del target_decoded, classes, classes_count

        n_samples = image_names.shape[0]
        shuffled_order = np.random.permutation(n_samples)
        image_names = image_names[shuffled_order]
        target = target[shuffled_order]

        p_train, p_valid = (isinstance(n, int) and n_samples or
                            int(n * n_samples)
                            for n in params['train_validation_test_split'])
        if p_train:
            self.train_image_names, self.train_data, self.train_target = (
                self._build_input_pipeline(image_names[:p_train],
                                           target[:p_train]))

        if p_valid and p_train < n_samples - 1:
            vn, vd, vt = self._build_input_pipeline(
                image_names[p_train:p_train + p_valid],
                target[p_train:p_train + p_valid])
            self.validation_image_names = vn
            self.validation_data = vd
            self.validation_target = vt

        if p_train + p_valid < n_samples - 1:
            self.test_image_names, self.test_data, self.test_target = (
                self._build_input_pipeline(image_names[p_train + p_valid:],
                                           target[p_train + p_valid:]))
        return self

    def _build_input_pipeline(self, image_names, target):
        params = self.parameters

        image, target = (tf.convert_to_tensor(image_names, dtype=tf.string),
                         tf.convert_to_tensor(target))
        image, target = tf.train.slice_input_producer(
            [image_names, target], num_epochs=params['n_epochs'])
        image = tf.image.decode_jpeg(tf.read_file(image), channels=3)

        return image_names, image, tf.cast(target, tf.float32)
