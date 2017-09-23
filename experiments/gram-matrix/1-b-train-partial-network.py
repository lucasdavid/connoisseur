"""1B Train Partial Network

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import random

import numpy as np
import tensorflow as tf
from keras import Input, layers, optimizers, callbacks, backend as K
from keras.engine import Model
from keras.models import Sequential
from keras.utils import to_categorical
from sacred import Experiment
from sklearn.preprocessing import StandardScaler

from connoisseur.datasets import load_pickle_data

ex = Experiment('1-b-train-partial-network')


@ex.config
def config():
    data_dir = '/mnt/files/datasets/pbn/'
    batch_size = 32
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    n_classes = 100
    ckpt_file = './ckpt/pbn,m%(id)s.hdf5'

    device = "/cpu:0"
    opt_params = {'lr': .001}

    epochs = 500
    initial_epoch = 0
    early_stop_patience = 30
    tensorboard_file = './logs/1-b-train-partial-network/vgg19,%s,batch-size:%i' % (opt_params, batch_size)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, indices):
    pairs = []
    labels = []
    # n = round(sum([len(indices[d]) for d in range(len(indices))]) / len(indices))
    n = max(map(len, indices))
    for d in range(len(indices)):
        count = len(indices[d])
        if not count: continue
        for i in range(n):
            z1, z2 = indices[d][i % count], indices[d][(i + 1) % count]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, len(indices))
            dn = (d + inc) % len(indices)
            while not len(indices[dn]):
                inc += 1
                dn = (d + inc) % len(indices)
            z1, z2 = indices[d][i % count], indices[dn][i % len(indices[dn])]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    return Sequential([
        layers.Dense(2048, activation='relu', input_shape=input_shape),
        layers.Dropout(0.5),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(0.5)
    ])


def compute_accuracy(predictions, labels):
    preds = predictions.ravel() < 0.5
    return ((preds & labels).sum() +
            (np.logical_not(preds) & np.logical_not(labels)).sum()) / float(labels.size)


@ex.automain
def main(data_dir, batch_size, style_layers, n_classes,
         ckpt_file, device, opt_params,
         epochs, initial_epoch,
         early_stop_patience, tensorboard_file):
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    n = tf.Session(config=tf_config)
    K.set_session(n)

    with tf.device(device):
        for m_id, layer_tag in enumerate(style_layers):
            data = load_pickle_data(data_dir, phases=('train', 'valid'),
                                    layers=[layer_tag], classes=n_classes)
            x, y, _ = data['train']
            xv, yv, _ = data['valid']
            x, xv = x[layer_tag], xv[layer_tag]
            x, xv = x.reshape(x.shape[0], -1), xv.reshape(xv.shape[0], -1)

            ss = StandardScaler()
            x = ss.fit_transform(x)
            xv = ss.transform(xv)

            input_shape = x.shape[1:]
            print('input shape:', input_shape)

            # labels = np.unique(y)
            # tr_indices = [np.where(y == l)[0] for l in labels]
            # vl_indices = [np.where(yv == l)[0] for l in labels]
            #
            # x, y = create_pairs(x, tr_indices)
            # xv, yv = create_pairs(xv, vl_indices)

            p = np.random.permutation(np.arange(x.shape[0]))
            x, y = x[p], y[p]
            del p

            y = to_categorical(y, num_classes=n_classes)
            yv = to_categorical(yv, num_classes=n_classes)

            # *** Softmax classifier ***
            # m = create_base_network(input_shape)
            i = Input(input_shape)
            n = layers.Dense(n_classes, activation='softmax')(i)
            m = Model(inputs=i, outputs=n)

            # *** Siamese distance regressor ***
            # a = Input(input_shape)
            # b = Input(input_shape)
            #
            # m = create_base_network(input_shape)
            # ya = m(a)
            # yb = m(b)
            #
            # n = layers.Lambda(euclidean_distance,
            #                   output_shape=lambda x: (x[0][0], 1))([ya, yb])
            # n = layers.BatchNormalization()(n)
            # n = layers.Dense(1, activation='sigmoid')(n)
            # m = Model(inputs=[a, b], outputs=n)

            m.compile(optimizer=optimizers.Adam(**opt_params),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

            print('training model #%i from epoch %i' % (m_id, initial_epoch))

            try:
                m.fit(
                    x, y,
                    # [x[:, 0], x[:, 1]], y,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, initial_epoch=initial_epoch,
                    validation_data=(xv, yv),  # ([xv[:, 0], xv[:, 1]], yv),
                    callbacks=[
                        # callbacks.LearningRateScheduler(lambda epoch: .5 ** (epoch // 10) * opt_params['lr']),
                        callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=int(early_stop_patience // 3)),
                        callbacks.EarlyStopping(patience=early_stop_patience),
                        callbacks.TensorBoard(tensorboard_file, histogram_freq=1, write_grads=True),
                        callbacks.ModelCheckpoint(ckpt_file % {'id': m_id}, save_best_only=True, verbose=1),
                    ])

            except KeyboardInterrupt:
                print('interrupted by user')
            else:
                print('done')

            # compute final accuracy on training and test sets
            # pred = m.predict([x[:, 0], x[:, 1]])
            # tr_acc = compute_accuracy(pred, y)
            # pred = m.predict([xv[:, 0], xv[:, 1]])
            # te_acc = compute_accuracy(pred, yv)
            # print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
            # print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

            del n, m
            K.clear_session()
