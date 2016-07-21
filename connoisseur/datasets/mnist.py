import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

from . import base


class MNIST(base.ImageDataSet):
    DEFAULT_PARAMETERS = {
        'batch_size': 50,
        'height': 28,
        'width': 28,
        'channels': 1,

        'train_validation_test_split': [.8, .2],

        'n_classes': 10,
        'save_in': '/tmp',
        'n_epochs': None,
        'source': 'http://yann.lecun.com/exdb/mnist/'
    }

    def __init__(self, **parameters):
        super().__init__('MNIST', **parameters)

    def load(self, override=False):
        params = self.parameters

        local_file = mnist.base.maybe_download('train-images-idx3-ubyte.gz', params['save_in'],
                                               params['source'] + 'train-images-idx3-ubyte.gz')
        self.train_data = mnist.extract_images(local_file)
        local_file = mnist.base.maybe_download('train-labels-idx1-ubyte.gz', params['save_in'],
                                               params['source'] + 'train-labels-idx1-ubyte.gz')
        self.train_target = mnist.extract_labels(local_file)

        p = np.random.permutation(self.train_data.shape[0])
        self.train_data = self.train_data[p]
        self.train_target = self.train_target[p]

        local_file = mnist.base.maybe_download('t10k-images-idx3-ubyte.gz', params['save_in'],
                                               params['source'] + 't10k-images-idx3-ubyte.gz')
        self.test_data = mnist.extract_images(local_file)
        local_file = mnist.base.maybe_download('t10k-labels-idx1-ubyte.gz', params['save_in'],
                                               params['source'] + 't10k-labels-idx1-ubyte.gz')
        self.test_target = mnist.extract_labels(local_file)
        p = np.random.permutation(self.test_data.shape[0])
        self.test_data = self.test_data[p]
        self.test_target = self.test_target[p]

        split_rates = params['train_validation_test_split']
        if split_rates[0] < 1:
            train_size = int(split_rates[0] * self.train_data.shape[0])

            self.validation_data = self.train_data[train_size:]
            self.validation_target = self.train_target[train_size:]
            self.train_data = self.train_data[:train_size]
            self.train_target = self.train_target[:train_size]

        for phase in ('train', 'validation', 'test'):
            images, labels = getattr(self, phase + '_data'), getattr(self, phase + '_target')
            if images is not None:
                images, labels = tf.train.slice_input_producer(
                    (tf.convert_to_tensor(images, dtype=tf.float32),
                     tf.convert_to_tensor(labels, dtype=tf.int32)),
                    num_epochs=1 if phase == 'test' else params['n_epochs'])

                setattr(self, phase + '_data', images)
                setattr(self, phase + '_target', labels)

        return self

    def download(self, override=False):
        return self

    def extract(self, override=False):
        return self
