"""Generate Salience for Paintings.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os

import keras.backend as K
import matplotlib
import numpy as np
import tensorflow as tf
from PIL import ImageFile
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from connoisseur import saliency, get_preprocess_fn
from connoisseur.models import build_model

matplotlib.use('agg')

from matplotlib import pylab as plt

ex = Experiment('generate-salience-for-paintings')

ex.captured_out_filter = apply_backspaces_and_linefeeds
ImageFile.LOAD_TRUNCATED_IMAGES = True


@ex.config
def config():
    image_shape = [299, 299, 3]
    data_dir = '/datasets/vangogh/vgdb_2016/train/'
    classes = ['vg']
    architecture = 'InceptionV3'
    weights = 'imagenet'
    last_base_layer = None
    use_gram_matrix = False
    pooling = 'avg'
    dense_layers = ()
    saliency_method = 'GuidedBackprop'

    device = '/cpu:0'
    dropout_p = 0.2
    ckpt_file = '/work/painter-by-numbers/ckpt/inception_softmax_auc:90.hdf5'

    output_index = 99  # van Gogh


def show_image(image, grayscale=True, ax=None, title=''):
    if ax is None:
        plt.figure()
    plt.axis('off')

    if len(image.shape) == 2 or grayscale:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)

        vmax = np.percentile(image, 99)
        vmin = np.min(image)

        plt.imshow(image, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.title(title)
    else:
        image = image + 127.5
        image = image.astype('uint8')

        plt.imshow(image)
        plt.title(title)


@ex.automain
def run(_run, image_shape, data_dir, classes,
        architecture, weights, last_base_layer,
        use_gram_matrix, pooling, dense_layers,
        saliency_method, device, dropout_p, ckpt_file,
        output_index):
    report_dir = _run.observers[0].dir

    tf.logging.set_verbosity(tf.logging.ERROR)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    preprocess_input = get_preprocess_fn(architecture)

    if not classes:
        classes = sorted(os.listdir(data_dir))

    with tf.device(device):
        print('building...')
        model = build_model(image_shape, architecture=architecture, weights=weights, dropout_p=dropout_p,
                            classes=1584, last_base_layer=last_base_layer,
                            use_gram_matrix=use_gram_matrix, pooling=pooling,
                            dense_layers=dense_layers)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam')

        if ckpt_file:
            print('re-loading weights...')
            model.load_weights(ckpt_file)

        saliency_cls = getattr(saliency, saliency_method)
        s = saliency_cls(model=model, output_index=output_index)

        try:
            for label in classes:
                samples = os.listdir(os.path.join(data_dir, label))
                os.makedirs(os.path.join(report_dir, label), exist_ok=True)

                for sample in samples:
                    x = img_to_array(load_img(os.path.join(data_dir, label, sample)))
                    x = preprocess_input(x)

                    # r = np.zeros(x.shape)
                    # mh, mw = [floor(x.shape[i] / image_shape[i]) * image_shape[i] + 1
                    #           for i in range(2)]
                    #
                    # cuts = itertools.product(range(0, mh, image_shape[0]),
                    #                          range(0, mw, image_shape[1]))
                    #
                    # for h, w in cuts:
                    #     p = x[h:h + image_shape[0], w:w + image_shape[0], :]
                    #     if p.shape != image_shape:
                    #         # Don't process borders. This can be  further
                    #         # improved by padding the input image.
                    #         continue
                    #
                    #     r[h:h + image_shape[0], w:w + image_shape[0], :] = s.get_mask(p)
                    r = s.get_mask(x)

                    print(r.min(), r.max())

                    name, extension = os.path.splitext(sample)

                    array_to_img(x).save(os.path.join(report_dir, label, name + '_o' + extension))
                    array_to_img(r).save(os.path.join(report_dir, label, name + '_s' + extension))
        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
