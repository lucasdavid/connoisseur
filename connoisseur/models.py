from keras import applications, layers, models, regularizers, backend as K, Input
from keras.engine import Model

from .utils import siamese_functions, gram_matrix


def get_base_model(architecture):
    """Finds an network inside one of the modules."""
    if architecture in globals():
        return globals()[architecture]

    if hasattr(applications, architecture):
        return getattr(applications, architecture)

    from keras_contrib.applications import densenet
    if hasattr(densenet, architecture):
        return getattr(densenet, architecture)

    raise ValueError('unknown architecture ' + architecture)


def build_model(image_shape, architecture, dropout_p=.5, weights='imagenet',
                classes=1000, last_base_layer=None, use_gram_matrix=False,
                dense_layers=(), pooling='avg', include_base_top=False, include_top=True,
                predictions_activation='softmax', predictions_name='predictions', model_name=None):
    base_model = get_base_model(architecture)(include_top=include_base_top, weights=weights,
                                              input_shape=image_shape, pooling=pooling)
    x = (base_model.get_layer(last_base_layer).output
         if last_base_layer
         else base_model.output)

    summary = '%s ' % architecture

    if use_gram_matrix:
        sizes = K.get_variable_shape(x)
        k = sizes[-1]
        x = layers.Lambda(gram_matrix, arguments=dict(norm_by_channels=False), output_shape=[k, k])(x)
        summary += '-> gram() '

    if include_top:
        if K.ndim(x) > 2:
            x = layers.Flatten(name='flatten')(x)
            summary += '-> flatten() '

        for l_id, n_units in enumerate(dense_layers):
            x = layers.Dense(n_units, activation='relu', name='fc%i' % l_id)(x)
            x = layers.Dropout(dropout_p)(x)
            summary += '-> dropout(relu(dense(%i))) ' % n_units

        if not isinstance(classes, (list, tuple)):
            classes, predictions_activation, predictions_name = ([classes],
                                                                 [predictions_activation],
                                                                 [predictions_name])

        outputs = []
        for units, activation, name in zip(classes, predictions_activation, predictions_name):
            outputs += [layers.Dense(units, activation=activation, name=name)(x)]
            summary += '\n  -> dense(%i, activation=%s, name=%s)' % (units, activation, name)
    else:
        outputs = [x]

    print('model summary:', summary)
    return Model(inputs=base_model.input, outputs=outputs, name=model_name)


def build_siamese_model(image_shape, architecture, dropout_rate=.5, weights='imagenet',
                        classes=1000, last_base_layer=None, use_gram_matrix=False,
                        dense_layers=(), pooling='avg', include_base_top=False,
                        include_top=True, limb_weights=None, predictions_activation='softmax',
                        trainable_limbs=True, embedding_units=1024, joint='multiply'):
    base_model = build_model(image_shape, architecture, dropout_rate, weights,
                             classes, last_base_layer, use_gram_matrix,
                             dense_layers, pooling, include_base_top,
                             include_top, predictions_activation)
    if limb_weights:
        print('loading weights from', limb_weights)
        base_model.load_weights(limb_weights)

    if not trainable_limbs:
        for l in base_model.layers:
            l.trainable = False

    if embedding_units:
        x = base_model.output
        x = layers.Dense(embedding_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(embedding_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(embedding_units, activation='relu')(x)
        base_model = Model(inputs=base_model.inputs, outputs=x)

    ia, ib = Input(shape=image_shape), Input(shape=image_shape)
    ya = base_model(ia)
    yb = base_model(ib)

    if joint == 'multiply':
        x = layers.multiply([ya, yb])
        x = layers.Dense(1, activation='sigmoid', name='binary_predictions')(x)
    else:
        if isinstance(joint, str):
            joint = siamese_functions[joint]
        x = layers.Lambda(joint, lambda _x: (_x[0][0], 1),
                          name='binary_predictions')([ya, yb])

    return Model([ia, ib], x)


def Inejc(include_top=False, weights=None, input_shape=(256, 256, 3), dropout_p=.5, pooling=None):
    weights_init_fn = 'he_normal'

    model = models.Sequential()

    model.add(_convolutional_layer(nb_filter=16, input_shape=input_shape))
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
