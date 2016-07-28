"""Paintings91 Connoisseur Training.


Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import json
import logging
import os

import connoisseur as conn
import tensorflow as tf

N_EPOCHS = 1000000

config = tf.ConfigProto(allow_soft_placement=True)

connoisseur_params = dict(
    n_epochs=N_EPOCHS, learning_rate=.001, dropout=.5,
    checkpoint_every=100,
    log_every=100,
    session_config=config)

data_set_params = dict(
    n_threads=1,
    train_validation_test_split=(.9, .1),
    save_in='./data/',
    batch_size=10,
    n_epochs=N_EPOCHS)


def main():
    os.makedirs(os.path.join(conn.settings.BASE_DIR, 'paintings91', 'logs'), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(conn.settings.BASE_DIR, 'paintings91', 'logs', 'training.log'),
        level=logging.DEBUG)

    logger = logging.getLogger('connoisseur')

    t = conn.utils.Timer()

    model = conn.connoisseurs.Paintings91(**connoisseur_params)
    dataset = conn.datasets.Paintings91(**data_set_params)

    logger.info('Executing with the following parameters:\n%s',
                json.dumps(dataset.parameters, indent=2))
    try:
        with tf.device('/cpu'):
            logger.info('fetching data set...')
            dataset.load().preprocess()
            training_data = dataset.as_batches()
            validation_data = dataset.as_batches(phase='validation')

        with tf.device('/gpu:1'):
            logger.info('training...')
            model.fit(*training_data, validation=validation_data)

            score = model.score(*validation_data)
            logger.info('score on validation dataset: %.2f%%', (100 * score))

    except KeyboardInterrupt:
        logger.info('interrupted by user (%s)', t)
    else:
        logger.info('finished (%s)', t)

    print('bye')


if __name__ == '__main__':
    print(__doc__, flush=True)
    main()
