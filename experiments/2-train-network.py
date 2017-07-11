"""Transfer and/or fine-tune models on a dataset.

Uses an architecture to train over a dataset.
Image patches are loaded directly from the disk.

Note: if you want to check the transfer learning results out, you may skip
this script and go straight ahead to "3-embed-patches.py".
However, make sure you are passing `pre_weights="imagenet"`.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

ex = Experiment('2-train-network')

ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    data_dir = "/datasets/vangogh/patches/random"
    batch_size = 128
    image_shape = [299, 299, 3]
    architecture = 'vgg19'
    weights = 'imagenet'
    last_base_layer = None
    use_gram_matrix = False
    dense_layers = (2048, 2048)
    train_shuffle = True
    dataset_train_seed = 12
    valid_shuffle = True
    dataset_valid_seed = 98
    device = "/gpu:0"

    n_classes = 2

    opt_params = {'lr': .001}
    dropout_p = 0.2
    resuming = False
    ckpt_file = './ckpt/opt-weights.hdf5'
    nb_epoch = 500
    steps_per_epoch = 78
    validation_steps = 10
    nb_worker = 8
    initial_epoch = 0
    early_stop_patience = 30
    tensorboard_file = './logs/2-train-network/%s,%s,batch-size:%i' % (architecture, opt_params, batch_size)
    first_trainable_layer = None
    first_reset_layer = None


@ex.automain
def run(image_shape, data_dir, train_shuffle, dataset_train_seed, valid_shuffle, dataset_valid_seed,
        n_classes,
        architecture, weights, batch_size, last_base_layer, use_gram_matrix, dense_layers,
        device, opt_params, dropout_p, resuming, ckpt_file, steps_per_epoch,
        nb_epoch, validation_steps, nb_worker, initial_epoch, early_stop_patience,
        tensorboard_file, first_trainable_layer, first_reset_layer):
    import os

    import tensorflow as tf
    from PIL import ImageFile
    from keras import callbacks, optimizers, backend as K
    from keras.preprocessing.image import ImageDataGenerator

    if architecture == 'vgg19':
        from keras.applications.vgg19 import preprocess_input
    else:
        from keras.applications.inception_v3 import preprocess_input

    from connoisseur.models import build_model

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    all_classes = os.listdir(os.path.join(data_dir, 'train'))
    classes = all_classes[:n_classes] if len(all_classes) > n_classes else None

    g = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        height_shift_range=.2,
        width_shift_range=.2,
        fill_mode='reflect',
        preprocessing_function=preprocess_input)

    train_data = g.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=image_shape[:2], classes=classes,
        batch_size=batch_size, shuffle=train_shuffle, seed=dataset_train_seed)

    val_data = g.flow_from_directory(
        os.path.join(data_dir, 'valid'),
        target_size=image_shape[:2], classes=classes,
        batch_size=batch_size, shuffle=valid_shuffle, seed=dataset_valid_seed)

    with tf.device(device):
        print('building...')
        model = build_model(image_shape, arch=architecture, weights=weights, dropout_p=dropout_p,
                            classes=n_classes, last_base_layer=last_base_layer,
                            use_gram_matrix=use_gram_matrix,
                            dense_layers=dense_layers)

        layer_names = [l.name for l in model.layers]

        if first_trainable_layer is not None:
            if first_trainable_layer not in layer_names:
                raise ValueError('%s is not a layer in the model: %s'
                                 % (first_trainable_layer, layer_names))

            _trainable = False
            for layer in model.layers:
                if layer.name == first_trainable_layer:
                    _trainable = True
                layer.trainable = _trainable
            del _trainable

        model.compile(optimizer=optimizers.Adam(**opt_params),
                      metrics=['accuracy'],
                      loss='categorical_crossentropy')

        if resuming:
            print('re-loading weights...')
            model.load_weights(ckpt_file)

        if first_reset_layer is not None:
            if first_reset_layer not in layer_names:
                raise ValueError('%s is not a layer in the model: %s'
                                 % (first_reset_layer, layer_names))
            print('first layer to have its weights reset:', first_reset_layer)
            random_model = build_model(image_shape, arch=architecture, weights=None, dropout_p=dropout_p,
                                       classes=n_classes, last_base_layer=last_base_layer,
                                       use_gram_matrix=use_gram_matrix,
                                       dense_layers=dense_layers)
            _reset = False
            for layer, random_layer in zip(model.layers, random_model.layers):
                if layer.name == first_reset_layer:
                    _reset = True
                if _reset:
                    layer.set_weights(random_layer.get_weights())
            del random_model

            model.compile(optimizer=optimizers.Adam(**opt_params),
                          metrics=['accuracy'],
                          loss='categorical_crossentropy')

        print('training from epoch %i...' % initial_epoch)
        try:
            model.fit_generator(
                train_data, steps_per_epoch=steps_per_epoch, epochs=nb_epoch, verbose=1,
                validation_data=val_data, validation_steps=validation_steps,
                workers=nb_worker,
                callbacks=[
                    callbacks.LearningRateScheduler(lambda epoch: .5 ** (epoch // 10) * opt_params['lr']),
                    callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=int(early_stop_patience // 3)),
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.TensorBoard(tensorboard_file),
                    callbacks.ModelCheckpoint(ckpt_file, save_best_only=True, verbose=1),
                ],
                initial_epoch=initial_epoch)

        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')