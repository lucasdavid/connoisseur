"""5 Evaluate Contrastive.

Evaluate trained InceptionV3 architecture onto van-Gogh test data set,
using the Contrastive fusion system.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

from time import time

import tensorflow as tf
from keras import backend as K
from sacred import Experiment

from connoisseur.datasets import VanGogh

from base import evaluate
from model import build_model

ex = Experiment('5-evaluate-contrastive')


@ex.config
def config():
    image_shape = [299, 299, 3]
    batch_size = 256
    device = '/gpu:0'
    data_dir = "/datasets/ldavid/van_gogh"
    ckpt_file = '/work/ldavid/van_gogh/5-train-contrastive/ckpt/optimal.hdf5'
    convolutions = True
    dataset_seed = 12
    train_n_patches = 40
    test_n_patches = 40
    n_jobs = 6


@ex.automain
def run(image_shape, batch_size, data_dir, ckpt_file, device,
        train_n_patches, test_n_patches, dataset_seed, n_jobs):
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    vangogh = VanGogh(
        base_dir=data_dir, image_shape=image_shape, random_state=dataset_seed,
        train_n_patches=train_n_patches, test_n_patches=test_n_patches,
        n_jobs=n_jobs)

    phases = ('train', 'test')

    print('loading data...')
    start = time()
    data = dict(zip(phases,
                    vangogh.load_patches_from_full_images('train', 'test')))
    print('done (%.1f sec).' % (time() - start))

    data['train'][0] /= 255.
    data['test'][0] /= 255.

    with tf.device(device):
        model = build_model(x_shape=image_shape, weights=None)
    model.load_weights(ckpt_file)
    scores = evaluate(model=model, data=data, batch_size=batch_size)
    print('max test score: %.2f%%' % (100 * scores['test']))
