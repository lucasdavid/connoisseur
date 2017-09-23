"""Generate Salience for Paintings.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import itertools
import os
from math import floor
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

import numpy as np
import tensorflow as tf

from PIL import Image, ImageFile
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K
from matplotlib import pylab as plt

from connoisseur.models import build_model
from connoisseur import saliency

ex = Experiment('generate-salience-for-paintings')

ex.captured_out_filter = apply_backspaces_and_linefeeds
ImageFile.LOAD_TRUNCATED_IMAGES = True


@ex.config
def config():
    image_shape = [299, 299, 3]
    data_dir = '/datasets/vangogh/vgdb_2016/train/'
    saliency_dir = '/datasets/vangogh/vgdb_2016/saliency/'
    classes = None
    architecture = 'InceptionV3'
    weights = 'imagenet'
    last_base_layer = None
    use_gram_matrix = False
    pooling = 'avg'
    dense_layers = ()
    saliency_method = 'GuidedBackprop'

    device = '/gpu:0'
    dropout_p = 0.2
    resuming = False
    ckpt_file = './ckpt/pbn,all-classes-,all-patches,inception.hdf5'


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
def run(image_shape, data_dir, classes,
        architecture, weights, last_base_layer, use_gram_matrix, pooling, dense_layers,
        saliency_method,
        device, dropout_p, resuming, ckpt_file):
    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
    os.makedirs(os.path.dirname(saliency_dir), exist_ok=True)

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

    if not classes:
        classes = sorted(os.listdir(data_dir))

    with tf.device(device):
        print('building...')
        model = build_model(image_shape, architecture=architecture, weights=weights, dropout_p=dropout_p,
                            classes=len(classes), last_base_layer=last_base_layer,
                            use_gram_matrix=use_gram_matrix, pooling=pooling,
                            dense_layers=dense_layers)

        if resuming:
            print('re-loading weights...')
            model.load_weights(ckpt_file)


        saliency_model = getattr(saliency, saliency_method)

        try:
            for label in classes:
                samples = os.listdir(os.path.join(data_dir, label))

                for sample in samples:
                    x = img_to_array(load_img(os.path.join(data_dir,
                                                           label,
                                                           sample)))
                    x = preprocess_input(x)
                    r = np.zeros(x.shape)

                    mh, mw = [floor(x.shape[i] / image_shape[i]) * image_shape[i] + 1
                              for i in range(2)]

                    cuts = itertools.product(range(0, mh, image_shape[0]),
                                             range(0, mw, image_shape[1]))

                    for h, w in cuts:
                        p = x[h:h + image_shape[0], w:w + image_shape[0], :]
                        r[h:h + image_shape[0], w:w + image_shape[0], :] = saliency_pipe.get_mask(p)

                    show_image(x), plt.show(), plt.hold()
                    show_image(r), plt.show(), plt.hold()
                    plt.clear()
                    plt.imshow(r)
                    plt.imsave(os.path.join(saliency_dir, label, sample))

        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
