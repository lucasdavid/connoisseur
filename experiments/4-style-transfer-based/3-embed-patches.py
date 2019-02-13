"""Embed Painting Patches.

This experiment consists on the following procedures:

 * Load each painting patch and transform it using a network.
 * Save the embeddings onto the disk.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import os
import pickle

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator
from sacred import Experiment

from connoisseur.models import build_gram_model
from connoisseur.utils import gram_matrix, get_preprocess_fn

ex = Experiment('embed-patches')

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
s = tf.Session(config=tf_config)
K.set_session(s)


@ex.config
def config():
    dataset_seed = 42
    batch_size = 256
    architecture = 'VGG19'
    weights = 'imagenet'
    image_shape = (299, 299, 3)
    device = "/gpu:0"
    data_dir = "/datasets/pbn/patches/random299"
    output_dir = data_dir
    phases = ['test']
    ckpt_file = '/work/pbn/gram/1/weights.hdf5'
    pooling = 'avg'
    override = False
    last_base_layer = None
    include_base_top = False
    include_top = False
    embedded_files_max_size = 5 * 1024 ** 3
    num_classes = 1483
    predictions_activation = 'softmax'
    dense_layers = []
    base_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1"
    ]
    selected_layers = ['predictions']


@ex.automain
def run(dataset_seed, image_shape, batch_size, device, data_dir, output_dir,
        phases, architecture, base_layers, predictions_activation,
        ckpt_file, weights, pooling, num_classes,
        dense_layers, override,
        embedded_files_max_size, selected_layers):
    os.makedirs(output_dir, exist_ok=True)

    with tf.device(device):
        print('building model...')
        model = build_gram_model(image_shape, architecture=architecture, weights=weights,
                                 classes=num_classes, base_layers=base_layers,
                                 pooling=pooling, dense_layers=dense_layers,
                                 predictions_activation=predictions_activation)
        if ckpt_file:
            # Restore best parameters.
            print('loading weights from:', ckpt_file)
            model.load_weights(ckpt_file)

        available_layers = [l.name for l in model.layers]
        if set(selected_layers) - set(available_layers):
            print('available layers:', available_layers)
            raise ValueError('selection contains unknown layers: %s' % selected_layers)

        style_features = [model.get_layer(l).output for l in selected_layers]
        model = Model(inputs=model.inputs, outputs=style_features)

    g = ImageDataGenerator(preprocessing_function=get_preprocess_fn(architecture))

    for phase in phases:
        phase_data_dir = os.path.join(data_dir, phase)
        output_file_name = os.path.join(output_dir, phase + '.%i.pickle')
        already_embedded = os.path.exists(output_file_name % 0)
        phase_exists = os.path.exists(phase_data_dir)

        if already_embedded and not override or not phase_exists:
            print('%s transformation skipped' % phase)
            continue

        # Shuffle must always be off in order to keep names consistent.
        data = g.flow_from_directory(phase_data_dir,
                                     target_size=image_shape[:2],
                                     class_mode='sparse',
                                     batch_size=batch_size, shuffle=False,
                                     seed=dataset_seed)
        print('transforming %i %s samples from %s' % (data.n, phase, phase_data_dir))
        part_id = 0
        samples_seen = 0
        displayed_once = False

        while samples_seen < data.n:
            z, y = {n: [] for n in selected_layers}, []
            chunk_size = 0
            chunk_start = samples_seen

            while chunk_size < embedded_files_max_size and samples_seen < data.n:
                _x, _y = next(data)

                outputs = model.predict_on_batch(_x)
                if not isinstance(outputs, list):
                    outputs = [outputs]

                chunk_size += sum(o.nbytes for o in outputs)

                for l, o in zip(selected_layers, outputs):
                    z[l].append(o)

                y.append(_y)
                samples_seen += _x.shape[0]
                chunk_p = int(100 * (samples_seen / data.n))

                if chunk_p % 10 == 0:
                    if not displayed_once:
                        print('\n%i%% (shape=%s, size=%.2f MB)'
                              % (chunk_p, _x.shape, chunk_size / 1024 ** 2),
                              flush=True, end='')
                        displayed_once = True
                else:
                    displayed_once = False
                    print('.', end='')

            for layer in selected_layers:
                z[layer] = np.concatenate(z[layer])

            with open(output_file_name % part_id, 'wb') as f:
                pickle.dump({'data': z,
                             'target': np.concatenate(y),
                             'names': np.asarray(data.filenames[chunk_start: samples_seen])},
                            f, pickle.HIGHEST_PROTOCOL)
            part_id += 1
    print('done.')
