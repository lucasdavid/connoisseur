from keras import backend as K
from keras.applications import InceptionV3
from keras.engine import Input, Model
from keras.layers import Lambda, Flatten, AveragePooling2D


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(
        K.maximum(margin - y_pred, 0)))


def build_model(x_shape, weights='imagenet'):
    base_network = InceptionV3(input_shape=x_shape, include_top=False,
                               weights=weights)
    x = base_network.output
    x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    base_network = Model(input=base_network.input, output=x)

    img_a = Input(shape=x_shape)
    img_b = Input(shape=x_shape)
    leg_a = base_network(img_a)
    leg_b = base_network(img_b)

    x = Lambda(euclidean_distance,
               output_shape=lambda x: (x[0][0], 1))([leg_a, leg_b])
    model = Model(input=[img_a, img_b], output=x)

    return model
