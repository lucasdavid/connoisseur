"""Generate Top Network Answer for Painter-by-Numbers.

This script takes the top network trained by 4-train-top-network.py script,
encodes each test paintings' pair into probabilities and save it on a file.
Finally, This file can be given to 6-generate-submission-file.py script.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import json
import os
from math import ceil

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import ImageFile
from keras import backend as K
from keras.preprocessing import image as ki
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from sacred import Experiment
from sklearn import metrics

from connoisseur import utils
from connoisseur.models import build_siamese_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
tf.logging.set_verbosity(tf.logging.ERROR)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
s = tf.Session(config=tf_config)
K.set_session(s)

ex = Experiment('generate-top-network-answer-for-painter-by-numbers')


@ex.config
def config():
    image_shape = [299, 299, 3]
    data_dir = '/datasets/pbn/random_299/'
    submission_info = '/datasets/pbn/submission_info.csv'
    solution = '/datasets/pbn/solution_painter.csv'
    num_classes = 1584
    predictions_activation = 'softmax'
    embedding_units = 1024
    architecture = 'InceptionV3'
    weights = 'imagenet'
    limb_weights = '/work/painter-by-numbers/ckpt/limb_weights.hdf5'
    patches = 50
    batch_size = 2  # seriously
    last_base_layer = None
    use_gram_matrix = False
    pooling = 'avg'
    embedding_units = 1024
    dense_layers = ()
    device = "/gpu:0"
    dropout_rate = 0.2
    ckpt = './ckpt/siamese.hdf5'
    results_file = './results.json'
    submission_file = 'answer-{strategy}.csv'
    estimator_type = 'score'

    use_multiprocessing = False
    workers = 1


def evaluate(labels, probabilities, estimator_type):
    print('aggregating patches')
    results = {
        'evaluations': [],
    }

    print('all done, proceeding to fusion')
    probabilities = probabilities.mean(axis=-1)

    if estimator_type == 'distance':
        # Closer distance means more likely the same (1.0)
        probabilities = 1 - probabilities

    probabilities = np.clip(probabilities, 0, 1)
    p = (probabilities > .5).astype(np.float)

    score = metrics.roc_auc_score(labels, probabilities)
    print('roc auc score using mean strategy:', score, '\n',
          metrics.classification_report(labels, p),
          '\nConfusion matrix:\n',
          metrics.confusion_matrix(labels, p))

    results['evaluations'].append({
        'strategy': 'mean',
        'score': score,
        'binary_probabilities': probabilities.tolist(),
        'p': p.tolist()
    })

    return results


class DirectoryPairsSequence(Sequence):
    def __init__(self, x, y, image_data_generator, target_size, batch_size: int, patches: int, base_dir: str):
        self.x, self.y = x, y
        self.image_data_generator = image_data_generator
        self.target_size = target_size
        self.batch_size = batch_size
        self.patches = patches
        if not base_dir.endswith('/'): base_dir += '/'
        self.base_dir = base_dir

    def __len__(self):
        return int(ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_names = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [[], []]
        # build batch of image data
        for a, b in batch_names:
            for i, n in enumerate((a, b)):
                n = self.base_dir + os.path.splitext(n)[0]
                for p in range(self.patches):
                    x = ki.img_to_array(ki.load_img('%s-%i.jpg' % (n, p), target_size=self.target_size))
                    x = self.image_data_generator.random_transform(x)
                    x = self.image_data_generator.standardize(x)
                    batch_x[i] += [x]

        return [np.asarray(_x) for _x in batch_x], np.repeat(batch_y, self.patches)

    def on_epoch_end(self):
        pass

            
def euclidean(inputs):
    x, y = inputs
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True),
                            K.epsilon()))


@ex.automain
def run(_run, image_shape, data_dir, patches, estimator_type, submission_info, solution, architecture, weights,
        batch_size, last_base_layer, use_gram_matrix, pooling, dense_layers, device, num_classes,
        limb_weights, predictions_activation, embedding_units, dropout_rate, ckpt, results_file, submission_file,
        use_multiprocessing, workers):
    report_dir = _run.observers[0].dir

    with tf.device(device):
        print('building...')
        model = build_siamese_model(image_shape, architecture, dropout_rate, weights, num_classes, last_base_layer,
                                    use_gram_matrix, dense_layers, pooling, include_base_top=False, include_top=True,
                                    predictions_activation=predictions_activation, limb_weights=limb_weights,
                                    trainable_limbs=False, embedding_units=embedding_units, joints='multiply')
        if ckpt:
            print('loading weights from', ckpt)
            model.load_weights(ckpt)

        print('loading submission and solution...')
        pairs = pd.read_csv(submission_info, quotechar='"', delimiter=',').values[:, 1:]
        labels = pd.read_csv(solution, quotechar='"', delimiter=',').values[:, 1:].flatten()

        print('\n# test evaluation')
        g = ImageDataGenerator(preprocessing_function=utils.get_preprocess_fn(architecture))
        data = DirectoryPairsSequence(pairs, labels, image_data_generator=g, target_size=image_shape,
                                      batch_size=batch_size, patches=patches, base_dir=data_dir + 'test/unknown/')
        probabilities = model.predict_generator(data, steps=len(data), verbose=1,
                                                use_multiprocessing=use_multiprocessing,
                                                workers=workers).reshape(-1, patches)
        del model
        K.clear_session()

    layer_results = evaluate(labels, probabilities, estimator_type)
    layer_results['phase'] = 'test'
    evaluation_results = [layer_results]

    # generate results file.
    with open(os.path.join(report_dir, results_file), 'w') as file:
        json.dump(evaluation_results, file)

    # generate submission file to Kaggle.
    for v in layer_results['evaluations']:
        predictions_field = 'binary_probabilities' if 'binary_probabilities' in v else 'p'
        p = v[predictions_field]

        with open(submission_file.format(strategy=v['strategy']), 'w') as f:
            f.write('index,sameArtist\n')
            f.writelines(['%i,%f\n' % (i, _p) for i, _p in enumerate(p)])
