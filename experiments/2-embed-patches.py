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

ex = Experiment('2-transform')


@ex.config
def config():
    dataset_seed = 4
    batch_size = 128
    architecture = 'inception'
    pre_weights = 'imagenet'
    image_shape = [299, 299, 3]
    device = "/cpu:0"
    data_dir = "/datasets/vangogh"
    pretrained_weights_path = None
    train_augmentations = []
    valid_augmentations = []
    test_augmentations = []
    override = False


@ex.automain
def run(dataset_seed, image_shape, batch_size, device, data_dir,
        train_augmentations, valid_augmentations, test_augmentations,
        architecture, pretrained_weights_path, pre_weights,
        override):
    import numpy as np
    import tensorflow as tf
    from keras import backend as K
    from keras.engine import Model

    from connoisseur.utils.image import ImageDataGenerator

    from connoisseur.models import build_model

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)
    os.makedirs('./results', exist_ok=True)

    with tf.device(device):
        print('building model...')
        # model = build_siamese_model(image_shape, arch=architecture, weights=pre_weights, dropout_p=0)
        model = build_model(image_shape, arch=architecture, weights=pre_weights, dropout_p=0)

        if pretrained_weights_path:
            # Restore best parameters.
            print('loading weights from:', pretrained_weights_path)
            model.load_weights(pretrained_weights_path)

        # model = model.layers[2]
        # extract_model = Model(input=model.get_input_at(0), output=model.get_layer('flatten').output)
        extract_model = Model(input=model.input, output=model.get_layer('flatten').output)

    g = ImageDataGenerator(rescale=2. / 255., featurewise_center=True)
    g.mean = 1

    for phase in ('train', 'valid', 'test'):
        phase_data_dir = os.path.join(data_dir, phase)
        output_file_name = os.path.join('./results/%s.pickle' % phase)

        if (os.path.exists(output_file_name) and not override or
                not os.path.exists(phase_data_dir)):
            print('%s transformation skipped' % phase)
            continue

        data = g.flow_from_directory(
            phase_data_dir,
            target_size=image_shape[:2], class_mode='categorical',
            augmentations=locals()[phase + '_augmentations'],
            batch_size=batch_size, shuffle=False, seed=dataset_seed)

        print('transforming %i %s samples' % (data.N, phase), end='')

        total_samples_seen = 0
        X, y = [], []

        while total_samples_seen < data.N:
            _X, _y = next(data)
            _X = extract_model.predict_on_batch(_X)

            X.append(_X)
            y.append(_y)
            total_samples_seen += _X.shape[0]
            if total_samples_seen % (batch_size * 20):
                print('.', end='', flush=True)
            else:
                print('%i%%' % round(100 * total_samples_seen / data.N),
                      end='', flush=True)

        if not X:
            continue

        X, y = np.concatenate(X), np.concatenate(y)
        with open(output_file_name, 'wb') as f:
            pickle.dump({'data': X, 'target': y, 'names': np.array(data.filenames, copy=False)},
                        f, pickle.HIGHEST_PROTOCOL)

    print('done.')
