"""
Paintings91 Classifier.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import time
import tensorflow as tf

import connoisseur as conn

data_set_params = dict(n_epochs=5,
                       batch_size=20,
                       n_threads=1,
                       save_in='/media/ldavid/hdd/data/')


class Paintings91Connoisseur(conn.Connoisseur):
    """Paintings91 Deep ConvolutionalNet Classifier."""

    def fit(self, data_set):
        started_at = time.time()

        ds_params = data_set.parameters

        with tf.Session() as s:
            images, labels = data_set.as_batches()

            dropout = tf.placeholder(tf.float32)

            if not self.network:
                # Initialize network with default VGG model + last fully
                # connected layer to 91 classes and a softmax classifier.
                m = conn.models.TwoLayers(images, labels, dropout)
                m.last_layer = conn.models.utils.fully_connect(
                    m.last_layer, *conn.models.utils.normal_layer([25 * 128, ds_params['n_classes']]))
                m.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(m.last_layer, labels))
                m.estimator = tf.nn.softmax(m.last_layer)

                self.network = m

            if not self.optimizer:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.network.loss)

            s.run(tf.initialize_all_variables())

            c = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=c)

            try:
                i = 0
                while not c.should_stop():
                    _, loss = s.run([self.optimizer, self.network.loss], {dropout: self.dropout})

                    print('[%i]: %.4f' % (i, loss))
                    i += 1

            except tf.errors.OutOfRangeError:
                print('Done, epoch limit reached (%.2f sec).'
                      % (time.time() - started_at))
            finally:
                c.request_stop()
            c.join(threads)

    def predict(self, X):
        with tf.Session() as s:
            return s.run(self.network.estimator, {'X': X, 'dropout': 1})


def main():
    print(__doc__)
    started_at = time.time()

    paintings91 = conn.datasets.Paintings91(**data_set_params)

    print('Executing with the following parameters: %s'
          % paintings91.parameters)

    try:
        print('Fetching data set...', end=' ')
        paintings91.load().preprocess()
        print('Done (%.2f sec).' % (time.time() - started_at))

        print('Training...')
        Paintings91Connoisseur().fit(paintings91)
        print('Done (%.2f sec).' % (time.time() - started_at))

    except KeyboardInterrupt:
        print('Canceled.')


if __name__ == '__main__':
    print(__doc__)
    main()
