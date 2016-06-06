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

from .. import settings

logger = logging.getLogger('connoisseur')


class PersistentEstimator:
    OUTER_SCOPE = None

    def __init__(self, checkpoint_every=20, checkpoints_dir='default'):
        if checkpoints_dir == 'default':
            checkpoints_dir = os.path.join(settings.BASE_DIR,
                                           self.__class__.__name__,
                                           'checkpoints')
        self.checkpoint_every = checkpoint_every
        self.checkpoints_dir = checkpoints_dir

        self._saver = self._optimal_saver = None
        self.loss_ = self.best_loss_ = np.inf
        self.epoch_ = 0

        # Create directory or confirm it exists.
        os.makedirs(os.path.join(checkpoints_dir, 'opt'), exist_ok=True)

    @property
    def saver(self):
        collection = tf.get_collection(tf.GraphKeys.VARIABLES,
                                       scope=self.OUTER_SCOPE)
        self._saver = self._saver or tf.train.Saver(collection)
        return self._saver

    @property
    def optimal_saver(self):
        collection = tf.get_collection(tf.GraphKeys.VARIABLES,
                                       scope=self.OUTER_SCOPE)
        self._optimal_saver = self._optimal_saver or tf.train.Saver(collection)
        return self._optimal_saver

    def save(self):
        """Persist the model whenever convenient.

        Persist a snapshot of the model every once in a while, as defined
        by the `checkpoint_every parameter`. Additionally, always save
        when loss is smaller than optimal.
        """
        if self.epoch_ % self.checkpoint_every == 0:
            self.saver.save(tf.get_default_session(),
                            os.path.join(self.checkpoints_dir, 'ckpt'),
                            global_step=self.epoch_)

        if self.loss_ < self.best_loss_:
            self.optimal_saver.save(
                tf.get_default_session(),
                os.path.join(self.checkpoints_dir, 'opt', 'ckpt'))

            self.best_loss_ = self.loss_

    def restore(self, checkpoint='last', raise_errors=True):
        """Restore a model persisted in disk.

        :param checkpoint: str, ['last', 'optimal']
            Which checkpoint to restore.
        :param raise_errors: bool, default=True.
            If true, raises an error whenever no checkpoint can be restored.
            Otherwise, silently goes over this step and has no effect.
        """
        checkpoint_dir = self.checkpoints_dir

        if checkpoint == 'optimal':
            checkpoint_dir = os.path.join(self.checkpoints_dir, 'opt')

        c = tf.train.get_checkpoint_state(checkpoint_dir)

        if c and c.model_checkpoint_path:
            # A checkpoint was found. Resume work.
            model = os.path.join(checkpoint_dir, c.model_checkpoint_path)
            logger.info('loading variables from {%s}', model)

            if checkpoint == 'last':
                self.saver.restore(tf.get_default_session(), model)
            elif checkpoint == 'optimal':
                self.optimal_saver.restore(tf.get_default_session(), model)
        elif raise_errors:
            raise ValueError('could not load variables from {%s}'
                             % checkpoint_dir)

        return self


class Connoisseur(PersistentEstimator, BaseEstimator, ClassifierMixin):
    """Connoisseur Base Class."""

    def __init__(self, n_epochs=100, learning_rate=0.001, dropout=.5,
                 checkpoint_every=20, checkpoints_dir='default',
                 logs_dir='default', session_config=None):
        PersistentEstimator.__init__(self, checkpoint_every,
                                     checkpoints_dir)
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.session_config = session_config

        if logs_dir == 'default':
            logs_dir = os.path.join(settings.BASE_DIR, self.__class__.__name__,
                                    'logs')
        self.logs_dir = logs_dir

        # Create directory or confirm it exits.
        os.makedirs(logs_dir, exist_ok=True)

    def _build(self, X, y=None, keep_prob=tf.constant(1.0), reuse=None):
        raise NotImplementedError

    def _build_fitted(self, X):
        s = tf.get_default_session()
        kp = tf.convert_to_tensor(1.0, name='keep_prob')

        try:
            # Called under the same execution of training.
            # Reuse everything.
            network = self._build(X, keep_prob=kp, reuse=True)
            s.run(tf.initialize_all_variables())
        except ValueError:
            # Create everything anew.
            network = self._build(X, keep_prob=kp)
            s.run(tf.initialize_all_variables())

            try:
                # Look optimal model state from the disk.
                self.restore(checkpoint='optimal')
            except ValueError:
                raise NotFittedError('This %s instance is not fitted yet.'
                                     ' Call `fit` with appropriate '
                                     'arguments before using this method.'
                                     % type(self).__name__)
        return network

    def fit(self, X, y, **fit_params):
        X, y, kp = (tf.convert_to_tensor(X, name='X'),
                    tf.convert_to_tensor(y, name='y'),
                    tf.convert_to_tensor(1.0 - self.dropout, name='keep_prob'))

        # Build inference model.
        network = self._build(X, y, kp, reuse=fit_params.get('reuse', None))

        validation = fit_params.get('validation', None)
        if validation:
            # Build validation model as a mirror of the inference model,
            # but using the validation data instead.
            X, y, kp = (tf.convert_to_tensor(validation[0], name='X_v'),
                        tf.convert_to_tensor(validation[1], name='y_v'),
                        tf.convert_to_tensor(1.0, name='keep_prob_v'))
            valid_network = self._build(X, y, kp, reuse=True)
            valid_loss = valid_network.loss
            valid_score = valid_network.score
        else:
            valid_loss = tf.constant(np.inf)
            valid_score = tf.constant(0.0)

        optimizer = (tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                     .minimize(network.loss))

        with tf.Session(config=self.session_config) as s:
            s.run(tf.initialize_all_variables())

            # Initiate summaries.
            tf.scalar_summary('loss', network.loss)
            tf.scalar_summary('validation-loss', valid_loss)
            merged_summaries = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(self.logs_dir,
                                                    graph=s.graph)
            try:
                self.restore(checkpoint='last')
            except ValueError:
                # We simply start our work again when no
                # previous saving points are found.
                logger.warning('could not restore persisted model. '
                               'Training will start from the beginning.')

            c = tf.train.Coordinator()
            ts = tf.train.start_queue_runners(coord=c)

            try:
                while self.epoch_ < self.n_epochs and not c.should_stop():
                    _, loss_, v_loss_, score_, v_score_, summary = s.run(
                        [optimizer,
                         network.loss, valid_loss,
                         network.score, valid_score,
                         merged_summaries])

                    self.loss_ = v_loss_ if validation else loss_
                    self.save()

                    # Log to Tensorboard.
                    summary_writer.add_summary(summary, self.epoch_)
                    # Log progress.
                    logger.info('[%i] loss: %.2f (%.2f), validation: %.2f (%.2f)',
                                self.epoch_, loss_, score_, v_loss_, v_score_)
                    self.epoch_ += 1

            except tf.errors.OutOfRangeError:
                logger.warning('input producer seems to be out of samples')
            finally:
                c.request_stop()
                c.join(ts)

        return self

    def predict(self, X):
        X = tf.convert_to_tensor(X, name='X')

        with tf.Session(config=self.session_config) as s:
            network = self._build_fitted(X)

            c = tf.train.Coordinator()
            ts = tf.train.start_queue_runners(coord=c)

            labels = []

            try:
                while not c.should_stop():
                    _likelihoods, _labels = s.run(network.estimator)
                    labels.append(_labels)
                    print(str(_labels.ravel()))
            except tf.errors.OutOfRangeError:
                logger.warning('input producer seems to be out of samples')
            finally:
                c.request_stop()
                c.join(ts)

            return np.concatenate(labels)

    def score(self, X, y, sample_weight=None):
        X, y = (tf.convert_to_tensor(X, name='X'),
                tf.convert_to_tensor(y, name='y'))

        with tf.Session(config=self.session_config) as s:
            s.run(tf.initialize_all_variables())
            network = self._build_fitted(X)

            c = tf.train.Coordinator()
            ts = tf.train.start_queue_runners(coord=c)

            score = cycles_ = 0

            try:
                while not c.should_stop():
                    labels_, batch_score_ = s.run([network.estimator[1],
                                                   network.score])
                    score += batch_score_
                    cycles_ += 1

            except tf.errors.OutOfRangeError:
                logger.info('all samples analyzed')

            finally:
                c.request_stop()
                c.join(ts)

            return score / cycles_
