import tensorflow as tf
from keras import optimizers, backend as K
from keras.applications import InceptionV3
from keras.engine import Input, Model
from keras.layers import Dense, Lambda, Dropout, Activation, Flatten, \
    AveragePooling2D
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


def build_model(x_shape, device='/gpu:0', opt_params={}, compile_opt=False,
                dropout_prob=.5, convolutions=False, freeze_base=False):
    with tf.device(device):
        if convolutions:
            base_network = InceptionV3(input_shape=x_shape)
            if freeze_base: base_network.trainable = False
            x = base_network.get_layer('flatten').output
            base_network = Model(input=base_network.input, output=x)
        else:
            base_network = Sequential([
                Dense(2048, activation='relu', input_shape=x_shape, name='fc1'),
                Dropout(dropout_prob),
                Dense(2048, activation='relu', name='fc2'),
                Dropout(dropout_prob),
                Dense(2048, activation='relu', name='fc3'),
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
