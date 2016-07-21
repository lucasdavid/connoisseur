"""MNIST Connoisseur Training.


Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import json
import logging
import os

import connoisseur as conn
import tensorflow as tf

N_EPOCHS = 1000

config = tf.ConfigProto(allow_soft_placement=True)

connoisseur_params = dict(
    n_epochs=N_EPOCHS, learning_rate=.001, dropout=.5,
    resume_training=True,
    log_every=100,
    checkpoint_every=1000,
    session_config=config)

data_set_params = dict(
    n_threads=1,
    save_in='./data/mnist',
    batch_size=50,
    train_validation_test_split=(.7, .3),
    n_epochs=N_EPOCHS)


def main():
    os.makedirs(os.path.join(conn.settings.BASE_DIR, 'mnist', 'logs'), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(conn.settings.BASE_DIR, 'mnist', 'logs', 'training.log'),
        level=logging.DEBUG)

    logger = logging.getLogger('connoisseur')

    t = conn.utils.Timer()

    model = conn.connoisseurs.MNIST(**connoisseur_params)
    dataset = conn.datasets.MNIST(**data_set_params)

    logger.info('Executing with the following parameters:\n%s',
                json.dumps(dataset.parameters, indent=2))

    try:
        with tf.device('/gpu:1'):
            logger.info('fetching data set...')
            dataset.load().preprocess()

            logger.info('training...')
            images, labels = dataset.as_batches('train')
            model.fit(images, labels, validation=dataset.as_batches('validation'))

            test_score = model.score(*dataset.as_batches('test'))
            logger.info('score on test dataset: %i%%', (100 * test_score))

    except KeyboardInterrupt:
        logger.info('interrupted by user (%s)', t)
    else:
        logger.info('finished (%s)', t)

    print('bye')


if __name__ == '__main__':
    print(__doc__, flush=True)
    main()
