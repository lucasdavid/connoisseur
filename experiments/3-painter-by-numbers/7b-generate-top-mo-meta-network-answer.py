"""Generate Top Network Answer for Painter-by-Numbers.

If the network trained in 4-train-top-network's limbs are too big, the test
activity will take too much time (I estimated 23 days using a Titan X).
In this case, encode the test data using 4-embed-patches.py and the limb
weights. Then use this script to load the trained siamese network, clip
its limbs and  test from the encoded data instead (it will take a few hours).

Encodes each test paintings' pair into probabilities and save it on a file.
Finally, This file can be given to 1-generate-submission-file.py script.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import ImageFile
from keras import Input, backend as K
from keras.engine import Model
from sacred import Experiment
from sklearn import metrics

from connoisseur.datasets import load_pickle_data, group_by_paintings
from connoisseur.models import build_siamese_mo_model
from connoisseur.utils.image import ArrayPairsSequence

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
    chunks = (0, 1)
    submission_info = '/datasets/pbn/submission_info.csv'
    solution = '/datasets/pbn/solution_painter.csv'
    architecture = 'InceptionV3'
    weights = 'imagenet'
    limb_weights = '/work/painter-by-numbers/ckpt/pbn_inception_mo.hdf5'
    limb_dense_layers = ()
    joint_weights = '/work/painter-by-numbers/ckpt/joint_weights.hdf5'
    patches = 50
    batch_size = 2  # seriously
    last_base_layer = None
    use_gram_matrix = False
    pooling = 'avg'
    dense_layers = (32,)
    device = "/gpu:0"
    dropout_rate = 0.2
    ckpt = '/work/painter-by-numbers/wlogs/train-top-top-mo/5/weights.hdf5'
    results_file = 'top-mo.json'
    submission_file = 'answer-{strategy}.csv'
    estimator_type = 'score'

    use_multiprocessing = False
    workers = 1

    outputs_meta = [
        dict(n='artist', u=1584, e=1024, j='multiply', a='softmax', l='binary_crossentropy', m='accuracy'),
        dict(n='style', u=135, e=256, j='multiply', a='softmax', l='binary_crossentropy', m='accuracy'),
        dict(n='genre', u=42, e=256, j='multiply', a='softmax', l='binary_crossentropy', m='accuracy'),
        # dict(n='date', u=1, e=256, j='l2', a='linear', l=utils.contrastive_loss, m=utils.contrastive_accuracy)
    ]


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


@ex.automain
def run(_run, image_shape, data_dir, patches, estimator_type, submission_info, solution, architecture, weights,
        batch_size, last_base_layer, use_gram_matrix, pooling, dense_layers, device, chunks,
        limb_weights, dropout_rate, ckpt, results_file, submission_file,
        use_multiprocessing, workers, outputs_meta, limb_dense_layers, joint_weights):
    report_dir = _run.observers[0].dir

    with tf.device(device):
        print('building...')
        model = build_siamese_mo_model(
            image_shape, architecture, outputs_meta,
            dropout_rate, weights,
            last_base_layer=last_base_layer,
            use_gram_matrix=use_gram_matrix,
            limb_dense_layers=limb_dense_layers,
            pooling=pooling,
            trainable_limbs=False,
            limb_weights=limb_weights,
            trainable_joints=False,
            joint_weights=joint_weights,
            dense_layers=dense_layers)

        print('loading weights from', ckpt)
        model.load_weights(ckpt)

        x = []
        for m in outputs_meta:
            name = m['n']
            shape = [m['e']]
            x += [Input(shape, name='%s_ia' % name), Input(shape, name='%s_ib' % name)]

        o = []
        for i, m in enumerate(outputs_meta):
            name = m['n']
            y = [x[2 * i], x[2 * i + 1]]
            y = model.get_layer('multiply_%i' % (i + 1))(y)
            y = model.get_layer('%s_binary_predictions' % name)(y)
            o += [y]

        rest = model.layers.index(model.get_layer('concatenate_asg'))
        for l in model.layers[rest:]:
            o = l(o)

        meta_model = Model(inputs=x, outputs=o)
        del model

        print('loading submission and solution...')
        pairs = pd.read_csv(submission_info, quotechar='"', delimiter=',').values[:, 1:]
        labels = pd.read_csv(solution, quotechar='"', delimiter=',').values[:, 1:].flatten()

        print('loading sequential predictions...')
        d = load_pickle_data(data_dir, phases=['test'], keys=['data', 'names'], chunks=chunks)
        samples, names = d['test']
        samples = np.asarray(list(zip(*(samples['%s_em3' % o['n']] for o in outputs_meta))))
        samples, names = group_by_paintings(samples, names=names)
        names = np.asarray([n.split('/')[1] + '.jpg' for n in names])

        print('test data shape:', samples.shape)

        print('\n# test evaluation')
        test_data = ArrayPairsSequence(samples, names, pairs, labels, batch_size)
        probabilities = meta_model.predict_generator(
            test_data,
            use_multiprocessing=use_multiprocessing,
            workers=workers, verbose=1).reshape(-1, patches)
        del meta_model
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

        with open(os.path.join(report_dir, submission_file.format(strategy=v['strategy'])), 'w') as f:
            f.write('index,sameArtist\n')
            f.writelines(['%i,%f\n' % (i, _p) for i, _p in enumerate(p)])
