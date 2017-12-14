"""Generate Top Network Answer for Painter-by-Numbers.

If the network trained in 4-train-top-network's limbs are too big, the test
activity will take too much time (I estimated 23 days using a Titan X).
In this case, encode the test data using 4-embed-patches.py and the limb
weights. Then use this script to load the trained siamese network, clip
its limbs and  test from the encoded data instead (it will take a few hours).

Encodes each test paintings' pair into probabilities and save it on a file.
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
from keras import Input, backend as K
from keras.engine import Model
from keras.utils import Sequence
from sacred import Experiment
from sklearn import metrics

from connoisseur.datasets import load_pickle_data, group_by_paintings
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
    data_dir = '/work/datasets/pbn/random_299/'
    submission_info = '/datasets/pbn/submission_info.csv'
    solution = '/datasets/pbn/solution_painter.csv'
    num_classes = 1584
    predictions_activation = 'softmax'
    embedding_units = 1024
    architecture = 'InceptionV3'
    weights = 'imagenet'
    limb_weights = '/work/painter-by-numbers/ckpt/limb_weights.hdf5'
    patches = 50
    batch_size = 10240  # seriously
    last_base_layer = None
    use_gram_matrix = False
    pooling = 'avg'
    dense_layers = ()
    device = "/gpu:0"
    dropout_rate = 0.2
    joint = 'multiply'
    ckpt = './ckpt/siamese.hdf5'

    estimator_type = 'probability'
    results_file = 'results.json'
    submission_file = 'answer-{strategy}.csv'

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

    roc_auc = metrics.roc_auc_score(labels, probabilities)
    print('roc auc: ', roc_auc, '\n',
          'accuracy: ', metrics.accuracy_score(labels, p, normalize=True), '\n',
          metrics.classification_report(labels, p), '\n',
          'confusion matrix:\n', metrics.confusion_matrix(labels, p),
          sep='')

    results['evaluations'].append({
        'strategy': 'mean',
        'score': roc_auc,
        'binary_probabilities': probabilities.tolist(),
        'p': p.tolist()
    })

    return results


class ArrayPairsSequence(Sequence):
    def __init__(self, samples, names, pairs, labels, batch_size):
        self.pairs = pairs
        self.labels = labels
        self.batch_size = batch_size
        self.samples_map = dict(zip(names, samples))
        self.samples, self.patches, self.features = samples.shape

    def __len__(self):
        return int(ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        xb = self.pairs[idx * self.batch_size:(idx + 1) * self.batch_size]
        yb = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        zb = np.concatenate([[self.samples_map[a], self.samples_map[b]] for a, b in xb], axis=1)
        yb = np.repeat(yb, self.patches)
        return list(zb), yb


@ex.automain
def run(_run, image_shape, data_dir, patches, estimator_type, submission_info, solution, architecture, weights,
        batch_size, last_base_layer, use_gram_matrix, pooling, dense_layers, device, num_classes,
        limb_weights, predictions_activation, joint, embedding_units, dropout_rate, ckpt,
        results_file, submission_file, use_multiprocessing, workers):
    report_dir = _run.observers[0].dir

    with tf.device(device):
        print('building...')
        model = build_siamese_model(image_shape, architecture, dropout_rate, weights, num_classes, last_base_layer,
                                    use_gram_matrix, dense_layers, pooling, include_base_top=False, include_top=True,
                                    predictions_activation=predictions_activation, limb_weights=limb_weights,
                                    trainable_limbs=False, embedding_units=embedding_units, joints=joint)
        model.summary()
        print('loading weights from', ckpt)
        model.load_weights(ckpt)

        limb, rest = model.get_layer('model_2'), model.layers[3:]
        x = i = Input(shape=(num_classes,))
        for l in limb.layers[-5:]:
            x = l(x)
        limb = Model(inputs=i, outputs=x)

        ia, ib = Input(shape=(num_classes,)), Input(shape=(num_classes,))
        ya = limb(ia)
        yb = limb(ib)

        x = [ya, yb]
        for l in rest:
            x = l(x)

        model = Model(inputs=[ia, ib], outputs=x)

        print('loading submission and solution...')
        pairs = pd.read_csv(submission_info, quotechar='"', delimiter=',').values[:, 1:]
        labels = pd.read_csv(solution, quotechar='"', delimiter=',').values[:, 1:].flatten()

        print('loading sequential predictions...')
        d = load_pickle_data(data_dir, phases=['test'], keys=['data', 'names'])
        d, names = d['test']
        samples = d['predictions']
        del d
        samples, _, names = group_by_paintings(samples, None, names)
        names = np.asarray([n.split('/')[1] + '.jpg' for n in names])

        print('test data shape:', samples.shape)

        print('\n# test evaluation')
        test_data = ArrayPairsSequence(samples, names, pairs, labels, batch_size)
        probabilities = model.predict_generator(
            test_data,
            use_multiprocessing=use_multiprocessing,
            workers=workers, verbose=1).reshape(-1, patches)
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
