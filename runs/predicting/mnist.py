"""Connoisseur Predicting MNIST.


Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import logging
import os

import connoisseur as conn
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)

connoisseur_params = dict(session_config=config)

dataset_params = dict(
    n_epochs=1,
    n_threads=1,
    train_validation_test_split=(.7, .3),
    save_in='../training/data/',
    batch_size=50)


def main():
    os.makedirs(os.path.join(conn.settings.BASE_DIR, 'mnist', 'logs'), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(conn.settings.BASE_DIR, 'mnist', 'logs', 'predicting.log'),
        level=logging.DEBUG)
    logger = logging.getLogger('connoisseur')

    t = conn.utils.Timer()

    model = conn.connoisseurs.MNIST(**connoisseur_params)
    dataset = conn.datasets.MNIST(**dataset_params)

    try:
        with tf.device('/gpu:1'):
            logger.info('fetching data set...')
            dataset.load().preprocess()

            images, labels = dataset.as_batches('train')
            score = model.score(images, labels)
            logger.info('score on training set: %i%%', int(100 * score))

            images, labels = dataset.as_batches('validation')
            score = model.score(images, labels)
            logger.info('score on validation set: %i%%', int(100 * score))

            images, labels = dataset.as_batches('test')
            score = model.score(images, labels)
            logger.info('score on test set: %i%%', int(100 * score))

    except KeyboardInterrupt:
        logger.info('interrupted by user (%s)', t)
    else:
        logger.info('finished (%s)', t)

    print('bye')


if __name__ == '__main__':
    print(__doc__, flush=True)
    main()
