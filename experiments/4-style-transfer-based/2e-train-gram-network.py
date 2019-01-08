"""Transfer and/or fine-tune models on a dataset.

Uses an architecture to train over a dataset.
Image patches are loaded directly from the disk.

Note: if you want to check the transfer learning results out, you may skip
this script and go straight ahead to "3-embed-patches.py".
However, make sure you are passing `pre_weights="imagenet"`.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os
from collections import Counter
from math import ceil

import tensorflow as tf
from PIL import ImageFile
from keras import callbacks, optimizers, backend as K
from keras.preprocessing.image import ImageDataGenerator
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from connoisseur.models import build_gram_model
from connoisseur.utils import get_preprocess_fn

ex = Experiment('train-gram-network')

ex.captured_out_filter = apply_backspaces_and_linefeeds
ImageFile.LOAD_TRUNCATED_IMAGES = True

tf.logging.set_verbosity(tf.logging.ERROR)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
s = tf.Session(config=tf_config)
K.set_session(s)


@ex.config
def config():
    data_dir = "/datasets/pbn/"
    batch_size = 64
    image_shape = [299, 299, 3]
    architecture = 'VGG16'
    weights = 'imagenet'
    base_layers = None
    dense_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1"
    ]
    pooling = 'avg'
    predictions_activation = 'softmax'
    train_shuffle = True
    dataset_train_seed = 12
    valid_shuffle = False
    dataset_valid_seed = 98
    device = "/gpu:0"

    classes = None

    metrics = ['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
    loss = 'sparse_categorical_crossentropy'
    opt_params = {'lr': .001}
    dropout_p = 0.4
    resuming_from_ckpt_file = None
    epochs = 500
    steps_per_epoch = None
    validation_steps = None
    workers = 8
    use_multiprocessing = False
    initial_epoch = 0
    early_stop_patience = 30
    tensorboard_tag = 'vangogh_%s' % architecture
    first_trainable_layer = None
    class_weight = 'balanced'
    class_mode = 'sparse'


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}


@ex.automain
def run(_run, image_shape, data_dir, train_shuffle, dataset_train_seed, valid_shuffle, dataset_valid_seed,
        classes, class_mode, class_weight,
        architecture, weights, batch_size, base_layers, pooling, dense_layers, predictions_activation,
        metrics, loss,
        device, opt_params, dropout_p, resuming_from_ckpt_file, steps_per_epoch,
        epochs, validation_steps, workers, use_multiprocessing, initial_epoch, early_stop_patience,
        tensorboard_tag, first_trainable_layer):
    report_dir = _run.observers[0].dir

    g = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        samplewise_center=True,
        samplewise_std_normalization=True,
        zoom_range=45,
        rotation_range=.2,
        height_shift_range=.2,
        width_shift_range=.2,
        fill_mode='reflect',
        preprocessing_function=get_preprocess_fn(architecture))

    if isinstance(classes, int):
        classes = sorted(os.listdir(os.path.join(data_dir, 'train')))[:classes]

    train_data = g.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=image_shape[:2], classes=classes, class_mode=class_mode,
        batch_size=batch_size, shuffle=train_shuffle, seed=dataset_train_seed)

    valid_data = g.flow_from_directory(
        os.path.join(data_dir, 'valid'),
        target_size=image_shape[:2], classes=classes, class_mode=class_mode,
        batch_size=batch_size, shuffle=valid_shuffle, seed=dataset_valid_seed)

    if class_weight == 'balanced':
        class_weight = get_class_weights(train_data.classes)

    if steps_per_epoch is None:
        steps_per_epoch = ceil(train_data.n / batch_size)
    if validation_steps is None:
        validation_steps = ceil(valid_data.n / batch_size)

    with tf.device(device):
        print('building...')
        model = build_gram_model(image_shape, architecture=architecture, weights=weights, dropout_p=dropout_p,
                                 classes=train_data.num_classes, base_layers=base_layers,
                                 pooling=pooling, dense_layers=dense_layers,
                                 predictions_activation=predictions_activation)
        model.summary()

        layer_names = [l.name for l in model.layers]

        if first_trainable_layer:
            if first_trainable_layer not in layer_names:
                raise ValueError('%s is not a layer in the model: %s'
                                 % (first_trainable_layer, layer_names))

            for layer in model.layers:
                if layer.name == first_trainable_layer:
                    break
                layer.trainable = False

        model.compile(optimizer=optimizers.Adam(**opt_params),
                      metrics=metrics,
                      loss=loss)

        if resuming_from_ckpt_file:
            print('re-loading weights...')
            model.load_weights(resuming_from_ckpt_file)

        print('training from epoch %i...' % initial_epoch)
        try:
            model.fit_generator(
                train_data, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=2,
                validation_data=valid_data, validation_steps=validation_steps,
                initial_epoch=initial_epoch,
                class_weight=class_weight,
                workers=workers, use_multiprocessing=use_multiprocessing,
                callbacks=[
                    # callbacks.LearningRateScheduler(lambda epoch: .5 ** (epoch // 10) * opt_params['lr']),
                    callbacks.TerminateOnNaN(),
                    callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=int(early_stop_patience // 3)),
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.TensorBoard(os.path.join(report_dir, tensorboard_tag), batch_size=batch_size),
                    callbacks.ModelCheckpoint(os.path.join(report_dir, 'weights.h5'), save_best_only=True, verbose=1),
                ])
        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
