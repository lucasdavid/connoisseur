"""4 Evaluate Triplet.

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

ex = Experiment('3-evaluate-triplet')


@ex.config
def config():
    image_shape = [299, 299, 3]
    batch_size = 256
    device = "/gpu:0"
    data_dir = "/datasets/ldavid/van_gogh"
    ckpt_file = './ckpt/opt-weights.hdf5'
    convolutions = True


@ex.automain
def run(image_shape, batch_size, data_dir, device, ckpt_file, convolutions):
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    data = {}
    for phase in ('train', 'valid', 'test'):
        try:
            pickle_file = os.path.join(data_dir, 'vgdb_2016',
                                       '%s.pickle' % phase)
            with open(pickle_file, 'rb') as f:
                data[phase] = pickle.load(f)
                data[phase] = (data[phase]['data'],
                               np.argmax(data[phase]['target'], axis=-1),
                               np.array(data[phase]['names'], copy=False))
        except IOError:
            continue

    model = build_model(x_shape=image_shape, convolutions=convolutions,
                        device=device)
    model.load_weights(ckpt_file)
    scores = evaluate(model=model, data=data, batch_size=batch_size)
    print('max test score: %.2f%%' % (100 * scores['test']))
