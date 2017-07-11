"""Embed Painting Patches' Gram Matrices.

This experiment consists on the following procedures:

 * Load each painting patch and transform it using a network.
 * Save the embeddings onto the disk.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import os
import pickle

from keras import backend as K
from sacred import Experiment

ex = Experiment('3-embed-patches-gram')


@ex.config
def config():
    dataset_seed = 4
    batch_size = 128
    architecture = 'vgg19'
    n_classes = 2
    pre_weights = 'imagenet'
    image_shape = [256, 256, 3]
    device = "/gpu:0"
    data_dir = "/datasets/vangogh"
    pretrained_weights_path = None
    override = False
    use_gram_matrix = False
    embedded_files_max_size = 10 * 1024 ** 3
    output_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']


@ex.automain
def run(dataset_seed, image_shape, batch_size, device, data_dir,
        architecture, n_classes, output_layers, pretrained_weights_path, pre_weights,
        use_gram_matrix,
        override, embedded_files_max_size):
    import numpy as np
    import tensorflow as tf
    from keras import layers
    from keras.engine import Model
    from keras.preprocessing.image import ImageDataGenerator
    from connoisseur.models import build_model
    from connoisseur.utils import gram_matrix

    if architecture == 'vgg19':
        from keras.applications.vgg19 import preprocess_input
    else:
        from keras.applications.inception_v3 import preprocess_input

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    with tf.device(device):
        print('building model...')
        # model = build_siamese_model(image_shape, arch=architecture, weights=pre_weights, dropout_p=0)
        model = build_model(image_shape, arch=architecture, weights=pre_weights, dropout_p=0,
                            classes=n_classes)

        if pretrained_weights_path:
            # Restore best parameters.
            print('loading weights from:', pretrained_weights_path)
            model.load_weights(pretrained_weights_path)

        # For siamese networks, get only first leg:
        # model = model.layers[2]
        # extract_model = Model(model.get_input_at(0), model.get_layer('flatten').output)
        
        style_features = [model.get_layer(l).output for l in output_layers]
        
        if use_gram_matrix:
            gram_layer = layers.Lambda(gram_matrix, arguments=dict(norm_by_channels=False))
            style_features = [gram_layer(f) for f in style_features]
        
        extract_models = [Model(model.input, f) for f in style_features]

    g = ImageDataGenerator(preprocessing_function=preprocess_input)

    for phase in ('train', 'valid', 'test'):
        phase_data_dir = os.path.join(data_dir, phase)
        output_file_name = os.path.join(data_dir, phase + '.%i.pickle')

        if (os.path.exists(output_file_name % 0) and not override or
                not os.path.exists(phase_data_dir)):
            print('%s transformation skipped' % phase)
            continue

        data = g.flow_from_directory(
            phase_data_dir,
            target_size=image_shape[:2], class_mode='sparse',
            batch_size=batch_size,
            # Shuffle must always be off in order to keep names consistent.
            shuffle=False, seed=dataset_seed)

        print('transforming %i %s samples' % (data.n, phase))

        part_id = 0
        samples_seen = 0

        while samples_seen < data.n:
            Z, y = {n: [] for n in output_layers}, []
            chunk_size = 0
            chunk_start = samples_seen
            displayed_once = False

            while chunk_size < embedded_files_max_size and samples_seen < data.n:
                _X, _y = next(data)

                for layer, m in zip(output_layers, extract_models):
                    _Z = m.predict_on_batch(_X)
                    Z[layer].append(_Z)

                y.append(_y)
                samples_seen += _Z.shape[0]
                chunk_percentage = int(100 * (samples_seen / data.n))
                chunk_size += _Z.nbytes

                if chunk_percentage % 10 == 0:
                    if not displayed_once:
                        print('%i%% (%.2f MB)...' % (chunk_percentage, chunk_size / 1024 ** 2),
                              flush=True, end=' ')
                        displayed_once = True
                else:
                    displayed_once = False

            for layer in output_layers:
                Z[layer] = np.concatenate(Z[layer])

            with open(output_file_name % part_id, 'wb') as f:
                pickle.dump({'data': Z,
                             'target': np.concatenate(y),
                             'names': np.array(data.filenames[chunk_start: samples_seen], copy=False)},
                            f, pickle.HIGHEST_PROTOCOL)
            part_id += 1
    print('done.')
