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

import tensorflow as tf
from PIL import ImageFile
from keras import callbacks, optimizers, backend as K
from keras.preprocessing.image import ImageDataGenerator
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from connoisseur.models import build_siamese_gram_model
from connoisseur.utils import get_preprocess_fn, contrastive_loss
from connoisseur.utils.image import BalancedDirectoryPairsSequence

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
    train_pairs = 1584
    valid_pairs = 1584
    image_shape = [299, 299, 3]

    architecture = 'VGG16'
    weights = 'imagenet'
    dense_layers = None
    base_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1"
    ]
    pooling = 'avg'
    device = "/gpu:0"

    classes = None

    metrics = ['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
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


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}


@ex.automain
def run(_run, image_shape, data_dir, train_pairs, valid_pairs,
        classes, class_weight,
        architecture, weights, batch_size, base_layers, pooling, dense_layers,
        metrics,
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

    train_data = BalancedDirectoryPairsSequence(os.path.join(data_dir, 'train'), g, target_size=image_shape[:2],
                                                pairs=train_pairs, classes=classes, batch_size=batch_size)
    valid_data = BalancedDirectoryPairsSequence(os.path.join(data_dir, 'valid'), g, target_size=image_shape[:2],
                                                pairs=valid_pairs, classes=classes, batch_size=batch_size)

    if class_weight == 'balanced':
        class_weight = get_class_weights(train_data.classes)

    with tf.device(device):
        print('building...')
        model = build_siamese_gram_model(image_shape, architecture, dropout_p, weights,
                                         base_layers=base_layers, dense_layers=dense_layers, pooling=pooling,
                                         include_top=False, trainable_limbs=True,
                                         embedding_units=0, joints='l2', include_base_top=False)
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
                      loss=contrastive_loss)

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
