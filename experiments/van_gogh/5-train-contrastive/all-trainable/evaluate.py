"""3 Test Contrastive.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import os
import pickle

import numpy as np
import tensorflow as tf
from keras import backend as K
from sacred import Experiment

from base import evaluate
from model import build_model

from connoisseur.datasets import VanGogh

ex = Experiment('3-test-contrastive')


@ex.config
def config():
    image_shape = [299, 299, 3]
    batch_size = 256
    device = "/gpu:0"
    data_dir = "/datasets/ldavid/van_gogh"
    ckpt_file = './ckpt/opt-weights.hdf5'
    convolutions = True
    dataset_seed = 12

    train_n_patches = 40
    test_n_patches = 40


@ex.automain
def run(image_shape, batch_size, data_dir, device, ckpt_file, convolutions,
        train_n_patches, test_n_patches, dataset_seed):
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    vangogh = VanGogh(
        base_dir=data_dir, image_shape=image_shape,
        train_n_patches=train_n_patches,
        test_n_patches=test_n_patches,
        random_state=dataset_seed
    ).download().extract().split_train_valid(valid_size=.1)

    phases = ('train', 'test')
    data = dict(zip(phases,
                    vangogh.load_patches_from_full_images('train', 'test')))

    model = build_model(x_shape=image_shape, convolutions=convolutions,
                        device=device)
    model.load_weights(ckpt_file)
    scores = evaluate(model=model, data=data, batch_size=batch_size)
    print('max test score: %.2f%%' % (100 * scores['test']))
