"""Train a network on top of the network trained on Painters-by-numbers.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os

import numpy as np
import tensorflow as tf
from PIL import ImageFile
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import OrderedEnqueuer
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn import metrics

from connoisseur import utils
from connoisseur.models import build_siamese_model
from connoisseur.utils.image import BalancedDirectoryPairsSequence

ex = Experiment('train-top-network')

ImageFile.LOAD_TRUNCATED_IMAGES = True
ex.captured_out_filter = apply_backspaces_and_linefeeds
tf.logging.set_verbosity(tf.logging.ERROR)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)
K.set_session(session)


@ex.config
def config():
    device = "/gpu:0"

    data_dir = "/datasets/vangogh/random_32/"
    valid_pairs = 1584
    num_classes = 1584
    classes = None

    batch_size = 128
    image_shape = [299, 299, 3]
    architecture = 'InceptionV3'
    weights = 'imagenet'
    last_base_layer = None
    use_gram_matrix = False
    dense_layers = ()
    embedding_units = 1024
    trainable_limbs = False
    pooling = 'avg'

    predictions_activation = 'softmax'
    limb_weights = '/work/painter-by-numbers/ckpt/limb_weights.hdf5'

    dropout_rate = 0.2
    ckpt = './model.hdf5'
    validation_steps = None
    use_multiprocessing = False


@ex.automain
def run(image_shape, data_dir, valid_pairs, classes,
        num_classes, architecture, weights, batch_size, last_base_layer, pooling, device, predictions_activation,
        dropout_rate, ckpt,
        validation_steps, use_multiprocessing, use_gram_matrix, dense_layers,
        embedding_units, limb_weights, trainable_limbs):
    if isinstance(classes, int):
        classes = sorted(os.listdir(os.path.join(data_dir, 'train')))[:classes]

    g = ImageDataGenerator(preprocessing_function=utils.get_preprocess_fn(architecture))
    valid_data = BalancedDirectoryPairsSequence(os.path.join(data_dir, 'valid'), g, target_size=image_shape[:2],
                                                pairs=valid_pairs, classes=classes, batch_size=batch_size)
    if validation_steps is None:
        validation_steps = len(valid_data)

    with tf.device(device):
        print('building...')
        model = build_siamese_model(image_shape, architecture, dropout_rate, weights, num_classes, last_base_layer,
                                    use_gram_matrix, dense_layers, pooling, include_base_top=False, include_top=True,
                                    predictions_activation=predictions_activation, limb_weights=limb_weights,
                                    trainable_limbs=trainable_limbs, embedding_units=embedding_units, joint='multiply')
        print('siamese model summary:')
        model.summary()
        if ckpt:
            print('loading weights...')
            model.load_weights(ckpt)

        enqueuer = None
        try:
            enqueuer = OrderedEnqueuer(valid_data, use_multiprocessing=use_multiprocessing)
            enqueuer.start()
            output_generator = enqueuer.get()

            y, p = [], []
            for step in range(validation_steps):
                x, _y = next(output_generator)
                _p = model.predict(x, batch_size=batch_size)
                y.append(_y)
                p.append(_p)

            y, p = (np.concatenate(e).flatten() for e in (y, p))

            print('actual:', y[:80])
            print('expected:', p[:80])
            print('accuracy:', metrics.accuracy_score(y, p >= 0.5))
            print(metrics.classification_report(y, p >= 0.5))
            print(metrics.confusion_matrix(y, p >= 0.5))

        finally:
            if enqueuer is not None:
                enqueuer.stop()
