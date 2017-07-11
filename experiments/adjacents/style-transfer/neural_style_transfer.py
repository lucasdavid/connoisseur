from __future__ import print_function

from keras import backend as K
from sacred import Experiment

ex = Experiment('neural-style-transfer')


@ex.config
def config():
    height = 512
    width = 512
    content_image_path = './contents/hugo.jpg'
    style_image_path = './style/wave.jpg'
    content_weight = 0.025
    style_weight = 5.0
    total_variation_weight = 1.0
    result_file_path = './result'


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination, height, width):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x, height, width):
    a = K.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
    b = K.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


@ex.automain
def main(width, height, content_image_path, style_image_path,
         content_weight, style_weight, total_variation_weight,
         result_file_path):
    import time
    from PIL import Image
    import numpy as np
    from keras.applications.vgg16 import VGG16
    from scipy.optimize import fmin_l_bfgs_b

    content_image = Image.open(content_image_path).resize((height, width))
    style_image = Image.open(style_image_path).resize((height, width))

    content_array = np.asarray(content_image, dtype='float32')
    content_array = np.expand_dims(content_array, axis=0)
    print(content_array.shape)

    style_array = np.asarray(style_image, dtype='float32')
    style_array = np.expand_dims(style_array, axis=0)
    print(style_array.shape)

    content_array[:, :, :, 0] -= 103.939
    content_array[:, :, :, 1] -= 116.779
    content_array[:, :, :, 2] -= 123.68
    content_array = content_array[:, :, :, ::-1]

    style_array[:, :, :, 0] -= 103.939
    style_array[:, :, :, 1] -= 116.779
    style_array[:, :, :, 2] -= 123.68
    style_array = style_array[:, :, :, ::-1]

    content_image = K.variable(content_array)
    style_image = K.variable(style_array)
    combination_image = K.placeholder((1, height, width, 3))

    input_tensor = K.concatenate([content_image,
                                  style_image,
                                  combination_image], axis=0)

    model = VGG16(input_tensor=input_tensor, weights='imagenet',
                  include_top=False)

    layers = dict([(layer.name, layer.output) for layer in model.layers])

    loss = K.variable(0.)

    def content_loss(content, combination):
        return K.sum(K.square(combination - content))

    layer_features = layers['block2_conv2']
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    loss += content_weight * content_loss(content_image_features,
                                          combination_features)

    feature_layers = ['block1_conv2', 'block2_conv2',
                      'block3_conv3', 'block4_conv3',
                      'block5_conv3']
    for layer_name in feature_layers:
        layer_features = layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, combination_features, height, width)
        loss += (style_weight / len(feature_layers)) * sl

    loss += total_variation_weight * total_variation_loss(combination_image,
                                                          height, width)

    grads = K.gradients(loss, combination_image)

    outputs = [loss]
    outputs += grads
    f_outputs = K.function([combination_image], outputs)

    def eval_loss_and_grads(x):
        x = x.reshape((1, height, width, 3))
        outs = f_outputs([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        return loss_value, grad_values

    class Evaluator(object):
        def __init__(self):
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = eval_loss_and_grads(x)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    evaluator = Evaluator()

    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

    iterations = 10

    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

    x = x.reshape((height, width, 3))
    x = x[:, :, ::-1]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')

    Image.fromarray(x).save(result_file_path)
