"""Connoisseur Base.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import NotFittedError

logger = logging.getLogger('connoisseur')


class TensorFlowClassifierMixin(ClassifierMixin):
    def score(self, X, y, sample_weight=None):
        # Hot decoding.
        _, y = tf.nn.top_k(y)

        with tf.Session() as s:

            s.run(tf.initialize_all_variables())

            c = tf.train.Coordinator()
            ts = tf.train.start_queue_runners(coord=c)

            try:
                X, y = s.run([X, y])
            finally:
                c.request_stop()
                c.join(ts)

        return super().score(X, y, sample_weight=sample_weight)


class Connoisseur(BaseEstimator, TensorFlowClassifierMixin):
    """Connoisseur Base Class."""

    LOGS = None
    CHECKPOINTS = None

    def __init__(self, n_epochs=100, learning_rate=0.001, dropout=.5):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.best_loss_ = np.inf

    def _build(self, X, y=None, dropout=tf.constant(0), reuse=None):
        raise NotImplementedError

    def fit(self, X, y, **fit_params):
        # Placeholders
        dropout_t = tf.placeholder(tf.float32, name='dropout_keep_prob')
        feed_dict = {dropout_t: self.dropout}

        # Build inference model.
        network = self._build(X, y,
                              dropout=dropout_t,
                              reuse=fit_params.get('reuse', None))

        if 'validation_data' in fit_params:
            # Build validation model as a mirror of the inference model,
            # but using the validation data instead.
            X_v, y_v = fit_params['validation_data']
            valid_loss = self._build(X_v, y_v,
                                     dropout=dropout_t,
                                     reuse=True).loss
        else:
            valid_loss = tf.constant(np.inf)

        optimizer = (tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                     .minimize(network.loss))

        with tf.Session() as s:
            s.run(tf.initialize_all_variables())

            # Initiate summaries.
            tf.scalar_summary('loss', network.loss)
            tf.scalar_summary('validation-loss', valid_loss)
            merged_summaries = tf.merge_all_summaries()

            summary_writer = tf.train.SummaryWriter(self.LOGS, graph=s.graph)
            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(self.CHECKPOINTS)
            if ckpt and ckpt.model_checkpoint_path:
                # A checkpoint was found. Resume work.
                logger.info('loading variables from {%s}',
                            ckpt.model_checkpoint_path)
                saver.restore(s, os.path.join(self.CHECKPOINTS,
                                              ckpt.model_checkpoint_path))

            c = tf.train.Coordinator()
            ts = tf.train.start_queue_runners(coord=c)

            try:
                epoch = 0

                while epoch < self.n_epochs and not c.should_stop():
                    _, train_loss_, valid_loss_, summary = s.run(
                        [optimizer, network.loss, valid_loss, merged_summaries],
                        feed_dict=feed_dict)

                    if epoch % 10 == 0:
                        # Save every 10 steps.
                        saver.save(s, self.CHECKPOINTS + '/ckpt',
                                   global_step=epoch)

                    if valid_loss_ < self.best_loss_:
                        f = os.path.join(self.CHECKPOINTS, 'opt', 'ckpt')
                        self.best_loss_ = valid_loss_

                    # Log to Tensorboard.
                    summary_writer.add_summary(summary, global_step=epoch)

                    # Log progress.
                    logger.info('[%i] training: %.2f, validation: %.2f',
                                epoch, train_loss_, valid_loss_)
                    epoch += 1

            except tf.errors.OutOfRangeError:
                logger.warning('input producer seems to be out of samples')
            finally:
                c.request_stop()
                c.join(ts)

        return self

    def predict(self, X):
        with tf.Session() as s:
            s.run(tf.initialize_all_variables())

            dropout_t = tf.placeholder(tf.dtypes.float32,
                                       name='dropout_keep_prob')
            feed_dict = {dropout_t: 0}

            input_data = X
            if isinstance(X, np.ndarray):
                input_data = tf.placeholder(X.dtype, X.shape)
                feed_dict[input_data] = X

            try:
                # Rebuild inference model from memory.
                network = self._build(input_data,
                                      dropout=dropout_t, reuse=True)
            except ValueError:
                # There's not trained model in memory. Look for it in the disk.
                saver = tf.train.Saver()
                opt_dir = os.path.join(self.CHECKPOINTS, 'opt')
                ckpt = tf.train.get_checkpoint_state(opt_dir)

                if ckpt and ckpt.model_checkpoint_path:
                    logger.info('loading variables from {%s}',
                                ckpt.model_checkpoint_path)
                    saver.restore(s, ckpt.model_checkpoint_path)
                else:
                    raise NotFittedError('This %s instance is not fitted yet. '
                                         'Call \'fit\' with appropriate '
                                         'arguments before using this method.'
                                         % type(self).__name__)

                network = self._build(input_data, dropout=dropout_t)

            c = tf.train.Coordinator()
            ts = tf.train.start_queue_runners(coord=c)

            try:
                likelihood, labels = s.run(network.estimator,
                                           feed_dict=feed_dict)
                return labels
            finally:
                c.request_stop()
                c.join(ts)
