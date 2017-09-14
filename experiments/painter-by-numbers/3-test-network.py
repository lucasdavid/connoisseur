"""Test fitted Network.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
from math import ceil
import numpy as np
from sacred import Experiment
from sklearn import metrics
from matplotlib import pylab as plt

ex = Experiment('3-test-network')


# ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    image_shape = [299, 299, 3]
    data_dir = "/datasets/vangogh/patches/"
    data_seed = 12
    classes = None
    architecture = 'InceptionV3'
    weights = 'imagenet'
    batch_size = 64
    last_base_layer = None
    use_gram_matrix = False
    pooling = 'avg'
    dense_layers = ()
    device = "/gpu:0"
    dropout_p = 0.2
    ckpt_file = './ckpt/pbn,all-classes-,all-patches,inception.hdf5'


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


def group_by_paintings(z, names):
    # Aggregate test patches by their respective paintings.
    _z, _names = [], []
    # Remove patches indices, leaving just the painting name.
    clipped_names = np.array(['-'.join(n.split('-')[:-1]) for n in names])
    for name in set(clipped_names):
        s = clipped_names == name
        _z.append(z[s][0])
        _names.append(clipped_names[s][0])

    return (np.array(_z, copy=False),
            np.array(_names, copy=False))


def evaluate(model, z, y, names, tag, group_patches, nb_patches, phase):
    score = metrics.accuracy_score(y, z)
    cm = metrics.confusion_matrix(y, z)
    print('score using raw strategy:', score, '\n',
          metrics.classification_report(y, z),
          '\nConfusion matrix:\n', cm)

    plot_confusion_matrix(cm, [str(i) for i in np.unique(y)],
                          name='-'.join((tag, phase, 'cm.jpg')))

    results = {
        'samples': names.tolist(),
        'labels': y.tolist(),
        'patches_count': nb_patches,
        'evaluations': [{
            'strategy': 'raw',
            'score': score,
            'p': p.tolist(),
        }]
    }

    if group_patches:
        z, names = group_by_paintings(z, names)
        for strategy_tag in ('sum', 'mean', 'farthest', 'most_frequent'):
            strategy = getattr(strategies, strategy_tag)

            p = (model, strategy=strategy).predict(x)
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
def run(image_shape, data_dir, data_seed, classes, architecture, weights,
        batch_size, last_base_layer, use_gram_matrix, pooling, dense_layers,
        device, dropout_p, ckpt_file):
    import os
    import tensorflow as tf
    from PIL import ImageFile
    from keras import backend as K
    from keras.preprocessing.image import ImageDataGenerator
    from connoisseur.models import build_model

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    # get appropriate pre-process function
    if architecture == 'InceptionV3':
        from keras.applications.inception_v3 import preprocess_input
    elif architecture == 'Xception':
        from keras.applications.xception import preprocess_input
    else:
        from keras.applications.imagenet_utils import preprocess_input

    n_classes = len(classes) if classes else len(os.listdir(os.path.join(data_dir, 'train')))

    g = ImageDataGenerator(preprocessing_function=preprocess_input)

    data = g.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=image_shape[:2], classes=classes,
        batch_size=batch_size, shuffle=False, seed=data_seed)



    with tf.device(device):
        print('building...')
        model = build_model(image_shape, architecture=architecture, weights=weights, dropout_p=dropout_p,
                            classes=n_classes, last_base_layer=last_base_layer,
                            use_gram_matrix=use_gram_matrix, pooling=pooling,
                            dense_layers=dense_layers)

        print('re-loading weights...')
        # model.load_weights(ckpt_file)

        names, z = [], []
        batches_to_see = int(ceil(data.n / data.batch_size))

        for b in range(batches_to_see):
            x, _ = next(data)
            names += data.filenames[batches_to_see:batches_to_see + batch_size]
            z.append(model.predict(x))

        z = np.concatenate(z)
        result = evaluate(z, y, names)
        z, names = group_by_paintings(z, names)

        print(z.shape)
