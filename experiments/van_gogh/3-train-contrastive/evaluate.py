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

ex = Experiment('3-test-contrastive')


@ex.config
def config():
    batch_size = 256
    device = "/gpu:0"
    data_dir = "/datasets/ldavid/van_gogh"
    ckpt_file = './ckpt/opt-weights.hdf5'


@ex.automain
def run(batch_size, data_dir, device, ckpt_file):
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

    model = build_model(x_shape=(2048,), device=device)
    model.load_weights(ckpt_file)
    evaluate(model=model, data=data, batch_size=batch_size)
