from keras import backend as K
from keras import models, regularizers
from keras.applications import InceptionV3
from keras.engine import Model, Input
from keras.layers import (BatchNormalization, PReLU, Dense, Conv2D,
                          Activation, Dropout, Flatten, MaxPooling2D, AveragePooling2D, Lambda)

L2_REG = 0.003
W_INIT = 'he_normal'
PENULTIMATE_SIZE = 2048
SOFTMAX_SIZE = 1584


def build_model(input_shape, arch='inception', dropout_p=.5, weights='imagenet'):
    assert arch in ('inception', 'inejc', 'siamese')

    return globals()['_arch_%s' % arch](input_shape=input_shape, weights=weights,
                                        dropout_p=dropout_p)


def build_siamese_model(x_shape, arch='inception', dropout_p=.5, weights='imagenet'):
    base_network = build_model(x_shape, arch=arch, dropout_p=dropout_p, weights=weights)
    base_network = Model(input=base_network.input,
                         output=base_network.get_layer('flatten').output)

    img_a = Input(shape=x_shape)
    img_b = Input(shape=x_shape)
    leg_a = base_network(img_a)
    leg_b = base_network(img_b)

    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    x = Lambda(euclidean_distance, output_shape=lambda x: (x[0][0], 1))([leg_a, leg_b])
    model = Model(input=[img_a, img_b], output=x)

    return model


def _arch_inception(input_shape, weights='imagenet', dropout_p=.5):
    base_model = InceptionV3(weights=weights, input_shape=input_shape, include_top=False)
    x = base_model.output
    x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(SOFTMAX_SIZE, activation='softmax', name='predictions')(x)

    return Model(input=base_model.input, output=x)


def _arch_inejc(input_shape, weights=None, dropout_p=.5):
    model = models.Sequential()

    model.add(_convolutional_layer(nb_filter=16, input_shape=input_shape))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=16))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(p=dropout_p))

    model.add(Flatten(name='flatten'))
    model.add(_dense_layer(output_dim=PENULTIMATE_SIZE))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT, name='prelu'))

    model.add(Dropout(p=dropout_p))
    model.add(_dense_layer(output_dim=SOFTMAX_SIZE))
    model.add(BatchNormalization(mode=2))
    model.add(Activation(activation='softmax', name='softmax'))

    return model


def _convolutional_layer(nb_filter, input_shape=None):
    if input_shape:
        return _first_convolutional_layer(nb_filter, input_shape)
    else:
        return _intermediate_convolutional_layer(nb_filter)


def _first_convolutional_layer(nb_filter, input_shape):
    return Conv2D(
        nb_filter=nb_filter, nb_row=3, nb_col=3, input_shape=input_shape,
        border_mode='same', init=W_INIT,
        W_regularizer=regularizers.l2(l=L2_REG))


def _intermediate_convolutional_layer(nb_filter):
    return Conv2D(
        nb_filter=nb_filter, nb_row=3, nb_col=3, border_mode='same',
        init=W_INIT, W_regularizer=regularizers.l2(l=L2_REG))


def _dense_layer(output_dim):
    return Dense(output_dim=output_dim,
                 W_regularizer=regularizers.l2(l=L2_REG),
                 init=W_INIT)
