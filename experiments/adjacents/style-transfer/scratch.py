import os

from keras.layers import Dense
from sacred import Experiment
from PIL import Image
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

import time

from keras import backend as K, Input
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

ex = Experiment('neural-style-collecting')


@ex.config
def config():
    data_dir = '/work/ldavid/datasets/vangogh/vgdb2016/train'
    input_shape = (512, 512, 3)
    target_shape = (512, 512, 3)
    n_epochs = 1

    content_weight = 0.025
    style_weight = 5.0
    total_variation_weight = 1.0


def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))

    x = np.asarray(image, dtype='float32')
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def content_loss(content, combination):
    return K.sum(K.square(combination - content))


@ex.automain
def main(data_dir, input_shape, target_shape,
         content_weight, style_weight, total_variation_weight,
         n_epochs):
    model = VGG16(weights='imagenet', include_top=True)

    layers = dict([(layer.name, layer.output) for layer in model.layers])

    print('vgg parameters count:', model.count_params())

    image_path = '/home/ldavid/Downloads/elephant.jpg'
    x = load_image(image_path)

    y = model.predict(x)
    print('predictions:\n', '\n'.join('%s:%f' % (e[1], e[2]) for e in decode_predictions(y)[0]))

    labels = os.listdir(data_dir)
    all_samples = [os.listdir(os.path.join(data_dir, l)) for l in labels]

    for samples, label in zip(all_samples, labels):
        print('processing label "%s"' % label)

        style_image = Input(input_shape)
        generated_image = Input(target_shape)

        layer_features = model.get_layer('block2_conv2')

        _model = Model(model.input, layer_features.output)

        content_image_features = layer_features.get_output_at(0)
        combination_features = layer_features.get_output_at(2)

        # loss = K.variable(0.)
        loss = content_weight * content_loss(content_image_features,
                                              combination_features)

        fmin_l_bfgs_b
