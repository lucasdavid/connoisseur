import tensorflow as tf
from keras import optimizers, backend as K
from keras.engine import Input, Model
from keras.layers import Dense, Lambda, Dropout, Activation
from keras.models import Sequential


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


def build_model(x_shape, device='/gpu:0', opt_params={}, compile_opt=False):
    with tf.device(device):
        base_network = Sequential([
            Dense(2048, input_shape=x_shape),
            Dropout(0.5),
            Activation('relu'),
            Dense(2048),
            Dropout(0.5),
            Activation('relu'),
            Dense(2048),
            Dropout(0.5),
            Activation('relu'),
        ])

        img_a = Input(shape=x_shape)
        img_b = Input(shape=x_shape)
        leg_a = base_network(img_a)
        leg_b = base_network(img_b)

        x = Lambda(euclidean_distance,
                   output_shape=lambda x: (x[0][0], 1))([leg_a, leg_b])
        model = Model(input=[img_a, img_b], output=x)

        if compile_opt:
            opt = optimizers.Adam(**opt_params)
            model.compile(optimizer=opt, loss=contrastive_loss)

    return model
