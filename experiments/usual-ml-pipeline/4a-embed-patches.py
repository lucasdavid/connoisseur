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

from connoisseur.models import build_model
from connoisseur.utils import gram_matrix, get_preprocess_fn

ex = Experiment('embed-patches')

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
s = tf.Session(config=tf_config)
K.set_session(s)


@ex.config
def config():
    dataset_seed = 4
    batch_size = 128
    architecture = 'InceptionV3'
    weights = 'imagenet'
    image_shape = (299, 299, 3)
    device = "/gpu:0"
    data_dir = "/datasets/vangogh-test-recaptures/recaptures-google/resized/patches/random/"
    output_dir = data_dir
    phases = ['test']
    ckpt_file = None
    pooling = 'avg'
    dense_layers = []
    override = False
    last_base_layer = None
    use_gram_matrix = False
    include_top = False
    embedded_files_max_size = 20 * 1024 ** 3
    o_meta = [
        dict(n='avg_pool', u=1584, e=1024, j='multiply', a='softmax', l='artist_predictions', m='accuracy'),
        dict(n='style', u=135, e=256, j='multiply', a='softmax', l='style_predictions', m='accuracy'),
        dict(n='genre', u=42, e=256, j='multiply', a='softmax', l='genre_predictions', m='accuracy'),
    ]

    selected_layers = ['global_average_pooling2d_1']


@ex.automain
def run(dataset_seed, image_shape, batch_size, device, data_dir, output_dir,
        phases, architecture, include_top,
        o_meta, ckpt_file, weights, pooling,
        dense_layers, use_gram_matrix, last_base_layer, override,
        embedded_files_max_size, selected_layers):
    os.makedirs(output_dir, exist_ok=True)

    with tf.device(device):
        print('building model...')
        model = build_model(image_shape, architecture=architecture,
                            weights=weights, dropout_p=.0,
                            pooling=pooling, last_base_layer=last_base_layer,
                            use_gram_matrix=use_gram_matrix,
                            dense_layers=dense_layers,
                            include_top=include_top,
                            classes=[o['u'] for o in o_meta],
                            predictions_name=[o['n'] for o in o_meta],
                            predictions_activation=[o['a'] for o in o_meta])
        if ckpt_file:
            # Restore best parameters.
            print('loading weights from:', ckpt_file)
            model.load_weights(ckpt_file)

        available_layers = [l.name for l in model.layers]
        if set(selected_layers) - set(available_layers):
            print('available layers:', available_layers)
            raise ValueError('selection contains unknown layers: %s' % selected_layers)

        style_features = [model.get_layer(l).output for l in selected_layers]

        if use_gram_matrix:
            gram_layer = layers.Lambda(gram_matrix, arguments=dict(norm_by_channels=False))
            style_features = [gram_layer(f) for f in style_features]

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
