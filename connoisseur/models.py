from keras import (applications, engine, layers, models, regularizers,
                   backend as K)
from keras.layers import GlobalAveragePooling2D

from .utils import euclidean, gram_matrix


def build_model(image_shape, arch='inception', dropout_p=.5, weights='imagenet',
                classes=1000, last_base_layer=None, use_gram_matrix=False,
                dense_layers=(), pooling='avg', include_top=True):
    assert arch in ('inception', 'inejc', 'vgg19')

    base_model = globals()['_arch_%s' % arch](image_shape, weights=weights, dropout_p=dropout_p,
                                              pooling=pooling)
    x = (base_model.get_layer(last_base_layer).output
         if last_base_layer
         else base_model.output)

    if use_gram_matrix:
        sizes = K.get_variable_shape(x)
        k = sizes[-1]
        x = layers.Lambda(gram_matrix, arguments=dict(norm_by_channels=False),
                          output_shape=[k, k])(x)

    if K.ndim(x) > 2:
        x = layers.Flatten(name='flatten')(x)

    if include_top:
        print('feature extraction network\'s output shape:', K.get_variable_shape(x))

        for n_units in dense_layers:
            print('dropout(relu(dense(x, %i))) layers stacked' % n_units)
            x = layers.Dense(n_units, activation='relu')(x)
            x = layers.Dropout(dropout_p)(x)

        x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    return engine.Model(base_model.input, x)


def build_siamese_model(x_shape, arch='inception', dropout_p=.5, weights='imagenet',
                        classes=1000, use_gram_matrix=False, distance=euclidean):
    base_network = build_model(x_shape, arch=arch, dropout_p=dropout_p, weights=weights,
                               classes=classes, use_gram_matrix=use_gram_matrix)
    base_network = engine.Model(base_network.input,
                                base_network.get_layer('flatten').output)

    ia = engine.Input(x_shape)
    ib = engine.Input(x_shape)

    ya, yb = map(base_network, (ia, ib))

    if isinstance(distance, str):
        distance = locals()[distance]

    x = layers.Lambda(distance, output_shape=lambda x: (x[0][0], 1))([ya, yb])
    model = engine.Model(input=[ia, ib], output=x)

    return model


def _arch_inception(input_shape, weights='imagenet', dropout_p=.5, pooling=None):
    return applications.InceptionV3(input_shape=input_shape, weights=weights,
                                    include_top=False, pooling=pooling)


def _arch_vgg19(input_shape, weights='imagenet', dropout_p=.5, pooling=None):
    return applications.VGG19(input_shape=input_shape, weights=weights,
                              include_top=False, pooling=pooling)


def _arch_inejc(batch_shape, weights=None, dropout_p=.5, pooling=None):
    weights_init_fn = 'he_normal'

    model = models.Sequential()

    model.add(_convolutional_layer(nb_filter=16, input_shape=batch_shape[1:]))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(_convolutional_layer(nb_filter=16))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=32))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(_convolutional_layer(nb_filter=32))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(_convolutional_layer(nb_filter=32))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=64))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(_convolutional_layer(nb_filter=64))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(_convolutional_layer(nb_filter=64))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=128))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(_convolutional_layer(nb_filter=128))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(_convolutional_layer(nb_filter=128))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=256))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(_convolutional_layer(nb_filter=256))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(_convolutional_layer(nb_filter=256))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(init=weights_init_fn))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(p=dropout_p))

    if pooling == 'avg':
        model.add(layers.GlobalAveragePooling2D())
    elif pooling == 'max':
        model.add(layers.GlobalMaxPooling2D())

    return model


def _convolutional_layer(nb_filter, input_shape=None):
    if input_shape:
        return _first_convolutional_layer(nb_filter, input_shape)
    else:
        return _intermediate_convolutional_layer(nb_filter)


def _first_convolutional_layer(nb_filter, input_shape,
                               l2_regularizer=.003, weights_init_fn='he_normal'):
    return layers.Conv2D(
        nb_filter=nb_filter, nb_row=3, nb_col=3, input_shape=input_shape,
        border_mode='same', init=weights_init_fn,
        W_regularizer=regularizers.l2(l=l2_regularizer))


def _intermediate_convolutional_layer(nb_filter, l2_regularizer=.003,
                                      weights_init_fn='he_normal'):
    return layers.Conv2D(
        nb_filter=nb_filter, nb_row=3, nb_col=3, border_mode='same',
        init=weights_init_fn, W_regularizer=regularizers.l2(l=l2_regularizer))


def _dense_layer(output_dim, l2_regularizer=.003, weights_init_fn='he_normal'):
    return layers.Dense(output_dim=output_dim,
                        W_regularizer=regularizers.l2(l=l2_regularizer),
                        init=weights_init_fn)
