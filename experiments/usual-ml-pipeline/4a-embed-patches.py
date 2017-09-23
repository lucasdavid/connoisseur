"""Embed Painting Patches.

This experiment consists on the following procedures:

 * Load each painting patch and transform it using a network.
 * Save the embeddings onto the disk.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import os
import pickle

from sacred import Experiment

ex = Experiment('embed-patches')


@ex.config
def config():
    dataset_seed = 4
    batch_size = 128
    architecture = 'InceptionV3'
    n_classes = None
    weights = 'imagenet'
    image_shape = (224, 224, 3)
    device = "/gpu:0"
    data_dir = "/datasets/vangogh"
    output_dir = data_dir
    phases = ['train', 'valid', 'test']
    ckpt_file = './ckpt/params.h5'
    override = False
    last_base_layer = None
    use_gram_matrix = False
    embedded_files_max_size = 20 * 1024 ** 3
    output_layers = ['mixed4']
    # ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']


@ex.automain
def run(dataset_seed, image_shape, batch_size, device, data_dir, output_dir, phases,
        architecture, n_classes, output_layers, ckpt_file, weights,
        use_gram_matrix, last_base_layer, override, embedded_files_max_size):
    import numpy as np
    import tensorflow as tf
    from keras import layers
    from keras.engine import Model
    from keras.preprocessing.image import ImageDataGenerator
    from keras import backend as K
    from connoisseur.models import build_model
    from connoisseur.utils import gram_matrix, get_preprocess_fn
    from connoisseur.utils.image import DirectoryIterator

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)
    os.makedirs(output_dir, exist_ok=True)

    with tf.device(device):
        print('building model...')
        # model = build_siamese_model(image_shape, arch=architecture, weights=weights, dropout_p=0)
        model = build_model(tuple(image_shape), architecture=architecture, weights=weights, dropout_p=0,
                            classes=n_classes, last_base_layer=last_base_layer,
                            use_gram_matrix=False, include_base_top=True, include_top=False,
                            dense_layers=())

        if ckpt_file:
            # Restore best parameters.
            print('loading weights from:', ckpt_file)
            model.load_weights(ckpt_file, by_name=True)

        # For siamese networks, get only first leg:
        # model = model.layers[2]
        # extract_model = Model(model.get_input_at(0), model.get_layer('flatten').output)

        print('available layers:', [l.name for l in model.layers])
        print('selected:', output_layers)

        style_features = [model.get_layer(l).output for l in output_layers]

        if use_gram_matrix:
            gram_layer = layers.Lambda(gram_matrix, arguments=dict(norm_by_channels=False))
            style_features = [gram_layer(f) for f in style_features]

        extract_models = [Model(model.input, f) for f in style_features]

    all_classes = sorted(os.listdir(os.path.join(data_dir, 'train')))
    classes = all_classes[:n_classes] if n_classes else None

    preprocess_input = get_preprocess_fn(architecture)

    g = ImageDataGenerator(preprocessing_function=preprocess_input)

    for phase in phases:
        phase_data_dir = os.path.join(data_dir, phase)
        output_file_name = os.path.join(output_dir, phase + '.%i.pickle')

        if os.path.exists(output_file_name % 0) and not override or not os.path.exists(phase_data_dir):
            print('%s transformation skipped' % phase)
            continue

        # Shuffle must always be off in order to keep names consistent.
        data = DirectoryIterator(phase_data_dir,
                                 classes=classes,
                                 image_data_generator=g,
                                 target_size=image_shape[:2], class_mode='sparse',
                                 batch_size=batch_size, shuffle=False, seed=dataset_seed)
        print('transforming %i %s samples' % (data.n, phase))

        part_id = 0
        samples_seen = 0
        displayed_once = False

        while samples_seen < data.n:
            z, y = {n: [] for n in output_layers}, []
            chunk_size = 0
            chunk_start = samples_seen

            while chunk_size < embedded_files_max_size and samples_seen < data.n:
                _x, _y = next(data)

                for layer, m in zip(output_layers, extract_models):
                    _z = m.predict_on_batch(_x)
                    z[layer].append(_z)
                    chunk_size += _z.nbytes

                y.append(_y)
                samples_seen += _x.shape[0]
                chunk_p = int(100 * (samples_seen / data.n))

                if chunk_p % 10 == 0:
                    if not displayed_once:
                        print('%i%% (%.2f MB)...' % (chunk_p, chunk_size / 1024 ** 2),
                              flush=True, end=' ')
                        displayed_once = True
                else:
                    displayed_once = False

            for layer in output_layers:
                z[layer] = np.concatenate(z[layer])

            with open(output_file_name % part_id, 'wb') as f:
                pickle.dump({'data': z,
                             'target': np.concatenate(y),
                             'names': np.array(data.filenames[chunk_start: samples_seen], copy=False)},
                            f, pickle.HIGHEST_PROTOCOL)
            part_id += 1
    print('done.')
