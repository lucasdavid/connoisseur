from keras import backend as K
from keras.engine import Input, Model
from keras.engine import InputLayer
from keras.layers import Dense, Dropout, Lambda
from keras.models import Sequential


def triplet_loss(y_true, y_pred):
    """Triplet Loss used in https://arxiv.org/pdf/1503.03832.pdf.
    """
    alpha = 1
    a, p, n = y_pred[:, :, 0], y_pred[:, :, 1], y_pred[:, :, 2]
    return K.sum(K.sqrt(K.sum((a - p) ** 2, axis=-1)) -
                 K.sqrt(K.sum((a - n) ** 2, axis=-1)) +
                 alpha)


def build_model(x_shape, dropout_prob=.5):
    b_net = Sequential([
        InputLayer(x_shape),
        Dense(2048, activation='relu', name='fc1'),
        Dropout(dropout_prob),
        Dense(2048, activation='relu', name='fc2'),
        Dropout(dropout_prob),
        Dense(2048, activation='relu', name='fc3'),
    ])

    img_a = Input(shape=x_shape)
    img_b = Input(shape=x_shape)
    img_c = Input(shape=x_shape)
    x = Lambda(lambda _x: K.concatenate(
        (K.expand_dims(_x[0]), K.expand_dims(_x[1]), K.expand_dims(_x[1]))
    ))([b_net(img_a), b_net(img_b), b_net(img_c)])

    t_net = Model(input=[img_a, img_b, img_c], output=x)

    return b_net, t_net
