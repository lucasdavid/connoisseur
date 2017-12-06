"""Generate Network Predictions.

Evaluate a network trained over a painting dataset.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import json
import os
from math import ceil

import matplotlib
import numpy as np
from PIL import ImageFile
from sacred import Experiment
from sklearn import metrics

matplotlib.use('agg')

from matplotlib import pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

ex = Experiment('generate-network-predictions')


@ex.config
def config():
    tag = 'random_299_inception_pca:0.99_svm'
    data_dir = '/datasets/vangogh-test-recaptures/patches/vangogh-museum-random-299/'
    n_classes = 2
    classes = None
    phases = ['test']
    data_seed = 19
    results_file_name = './results-%s.json' % tag
    group_patches = True
    batch_size = 64
    image_shape = [299, 299, 3]
    architecture = 'DenseNet'
    last_base_layer = None
    use_gram_matrix = False
    dense_layers = ()
    pooling = 'avg'
    weights = 'imagenet'
    dropout_p = 0.2
    ckpt_file = '/work/vangogh/models/vangogh_densenet.hdf5'
    device = "/gpu:0"


def plot_confusion_matrix(cm, labels, name='cm.jpg', **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, **kwargs)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fig.savefig(name)


def evaluate(probabilities, y, names, tag, group_patches, phase):
    from connoisseur.datasets import group_by_paintings
    from connoisseur.fusion import Fusion, strategies

    p = np.argmax(probabilities, axis=-1)
    score = metrics.accuracy_score(y, p)
    cm = metrics.confusion_matrix(y, p)
    print('score using raw strategy:', score, '\n',
          metrics.classification_report(y, p),
          '\nConfusion matrix:\n', cm)

    plot_confusion_matrix(cm, [str(i) for i in np.unique(y)],
                          name='-'.join((tag, phase, 'cm.jpg')),
                          cmap='BuPu')

    results = {
        'samples': names,
        'labels': y.tolist(),
        'evaluations': [{
            'strategy': 'raw',
            'score': score,
            'p': p.tolist(),
        }]
    }

    if group_patches:
        probabilities, y, names = group_by_paintings(probabilities, y, names)
        for strategy_tag in ('mean', 'farthest', 'most_frequent'):
            strategy = getattr(strategies, strategy_tag)

            p = Fusion(strategy=strategy).predict(probabilities)
            score = metrics.accuracy_score(y, p)
            print('score using', strategy_tag, 'strategy:', score, '\n',
                  metrics.classification_report(y, p),
                  '\nConfusion matrix:\n',
                  metrics.confusion_matrix(y, p))
            print('samples incorrectly classified:', names[p != y], '\n')

            results['evaluations'].append({
                'strategy': strategy_tag,
                'score': score,
                'p': p.tolist()
            })

    return results


@ex.automain
def run(tag, data_dir, n_classes, phases, classes, data_seed,
        results_file_name, group_patches, batch_size, image_shape,
        architecture, weights, dropout_p, last_base_layer,
        use_gram_matrix, pooling, dense_layers,
        device, ckpt_file):
    import tensorflow as tf
    from keras import backend as K
    from keras.preprocessing.image import ImageDataGenerator
    from connoisseur.models import build_model
    from connoisseur.utils import get_preprocess_fn

    tf.logging.set_verbosity(tf.logging.ERROR)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    preprocess_input = get_preprocess_fn(architecture)

    g = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        preprocessing_function=None)

    with tf.device(device):
        print('building...')
        model = build_model(image_shape, architecture=architecture, weights=weights, dropout_p=dropout_p,
                            classes=n_classes, last_base_layer=last_base_layer,
                            use_gram_matrix=use_gram_matrix, pooling=pooling,
                            dense_layers=dense_layers)

        if ckpt_file:
            print('re-loading weights...')
            model.load_weights(ckpt_file)

        results = []
        for phase in phases:
            print('\n# %s evaluation' % phase)

            data = g.flow_from_directory(
                os.path.join(data_dir, phase),
                target_size=image_shape[:2], classes=classes,
                batch_size=batch_size, seed=data_seed,
                shuffle=False,
                class_mode='sparse')

            steps = ceil(data.n / batch_size)

            probabilities = model.predict_generator(data, steps=steps)
            layer_results = evaluate(probabilities=probabilities, y=data.classes, names=data.filenames,
                                     tag=tag, group_patches=group_patches, phase=phase)
            layer_results['phase'] = phase
            results.append(layer_results)

    with open(results_file_name, 'w') as file:
        json.dump(results, file)
