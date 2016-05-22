import os

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

from . import base


class Paintings91(base.ImageDataSet):
    DEFAULT_PARAMETERS = {
        'file_name': 'Paintings91.zip',
        'source': 'http://cat.cvc.uab.es/~joost/data/Paintings91.zip',
        'expected_size': 681584272,
        'batch_size': 50,
        'height': 400,
        'width': 400,
        'n_samples_per_epoch': 10,
        'n_threads': 2,
        'min_samples_in_queue': 0.4,
        'n_classes': 91,
        'save_in': '/tmp',

        'n_epochs': None,
    }

    def __init__(self, parameters=None):
        # Paintings91 already know its name.
        super().__init__('Paintings91', parameters)

    def load(self, override=False):
        self.download(override=override).extract(override=override)

        params = self.parameters

        base_folder = os.path.join(params.save_in, self.name, 'Paintings91')

        if not os.path.exists(base_folder):
            raise RuntimeError('Data set not found. Have you downloaded and '
                               'extracted it first?')

        images_folder = os.path.join(base_folder, 'Images')
        labels_folder = os.path.join(base_folder, 'Labels', 'labels.mat')

        # Data is represented at a column vector. Coverts it to a list.
        image_names = loadmat(os.path.join(base_folder, 'Labels',
                                           'image_names.mat'))['image_names']
        image_names = [os.path.join(images_folder, file_name[0][0])
                       for file_name in image_names]

        # Be aware that this target is one-hot encoded already.
        target = (loadmat(labels_folder)['labels'])

        self.image_names = image_names

        image, target = (
            tf.convert_to_tensor(image_names, dtype=tf.string),
            tf.convert_to_tensor(target),
        )

        image, target = (tf.train
                         .slice_input_producer([image_names, target],
                                               num_epochs=params.n_epochs))
        image = tf.image.decode_jpeg(tf.read_file(image), channels=3)
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        self.data = image
        self.target = target

        return self
