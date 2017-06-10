"""5 Evaluate Contrastive.

Evaluate trained InceptionV3 architecture onto van-Gogh test data set,
using the Contrastive fusion system.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

from time import time

import numpy as np
import tensorflow as tf
from keras import backend as K
from sacred import Experiment

from connoisseur.datasets import VanGogh
from connoisseur.models import build_siamese_model

from base import evaluate

ex = Experiment('5-evaluate-contrastive')


@ex.config
def config():
    image_shape = [256, 256, 3]
    batch_size = 40
    device = '/gpu:0'
    data_dir = "/datasets/vangogh"
    ckpt_file = '/work/vangogh/5-train-contrastive/ckpt/optimal.hdf5'
    dataset_seed = 12
    train_n_patches = 50
    test_n_patches = 50
    patches_used_in_eval = 40
    arch = 'inejc'
    n_jobs = 6


@ex.automain
def run(image_shape, batch_size, device, data_dir,
        ckpt_file, dataset_seed, train_n_patches, test_n_patches,
        patches_used_in_eval, arch, n_jobs):
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    vangogh = VanGogh(
        base_dir=data_dir, image_shape=image_shape, random_state=dataset_seed,
        train_n_patches=train_n_patches, valid_n_patches=train_n_patches, test_n_patches=test_n_patches,
        n_jobs=n_jobs)

    phases = ('train', 'valid', 'test')

    print('loading data...')
    start = time()
    data = dict(zip(phases, vangogh.load_patches(*phases)))
    print('done (%.1f sec).' % (time() - start))

    data['train'] = list(map(np.concatenate, zip(data['train'], data['valid'])))
    del data['valid']

    data['train'][0] /= 127.5
    data['train'][0] -= 1.
    data['test'][0] /= 127.5
    data['test'][0] -= 1.

    with tf.device(device):
        model = build_siamese_model(x_shape=image_shape, arch=arch, weights=None)
        model.load_weights(ckpt_file)
        scores = evaluate(model=model, data=data, batch_size=batch_size,
                          patches_used=patches_used_in_eval)
        print('max test score: %.2f%%' % (100 * scores['test']))
