"""Visualizing Extracted Patches.

Extract patches from the paintings using the ImageDataGenerator Keras'
backdoor in connoisseur module. This experiment is only made to confirm
that the extraction happens as expected.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import logging
import os
import shutil

import numpy as np
import tensorflow as tf
from artificial.utils.experiments import ExperimentSet, Experiment, arg_parser
from keras import backend as K

from connoisseur import datasets
from connoisseur.utils.image import ImageDataGenerator


class VisualizingExtractedPatchesExperiment(Experiment):
    def setup(self):
        c = self.consts
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        s = tf.Session(config=config)
        K.set_session(s)
        np.random.seed(c.seed)

    def run(self):
        c = self.consts

        tf.logging.debug('checking van-Gogh data set...')
        datasets.VanGogh(c).download().extract().check()

        g = ImageDataGenerator()

        train_data = g.flow_from_directory(
            os.path.join(c.data_dir, 'vgdb_2016', 'train'),
            target_size=c.image_shape[:2],
            extraction_method='random-crop',
            batch_size=c.batch_size,
            shuffle=c.train_shuffle,
            seed=c.dataset_seed,
            save_prefix='train',
            save_to_dir='./tmp/train')

        test_data = g.flow_from_directory(
            os.path.join(c.data_dir, 'vgdb_2016', 'test'),
            target_size=c.image_shape[:2],
            extraction_method='random-crop',
            batch_size=c.batch_size,
            shuffle=c.val_shuffle,
            seed=c.dataset_val_seed,
            save_to_dir='./tmp/test',
            save_prefix='test')

        os.makedirs('./results/train', exist_ok=True)
        os.makedirs('./results/test', exist_ok=True)

        for epoch in range(c.n_epochs):
            os.mkdir('./tmp/train')
            os.mkdir('./tmp/test')

            samples_seen = 0
            while samples_seen < c.samples_per_epoch:
                next(train_data)
                samples_seen += c.batch_size
            samples_seen = 0
            while samples_seen < c.nb_val_samples:
                next(test_data)
                samples_seen += c.batch_size
            shutil.move('./tmp/train', './results/train/%i' % epoch)
            shutil.move('./tmp/test', './results/test/%i' % epoch)

    def teardown(self):
        K.clear_session()
        os.rmdir('./tmp')


if __name__ == '__main__':
    args = arg_parser.parse_args()

    print(__doc__, flush=True)

    logging.basicConfig(level=logging.INFO, filename='./run.log')
    for logger in ('artificial', 'tensorflow'):
        logger = logging.getLogger(logger)
        logger.setLevel(logging.DEBUG)

    (ExperimentSet(experiment_cls=VisualizingExtractedPatchesExperiment)
     .load_from_json(args.constants)
     .run())
