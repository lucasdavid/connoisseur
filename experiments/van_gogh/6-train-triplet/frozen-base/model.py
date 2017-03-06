import tensorflow as tf
import tensorflow.contrib.slim as slim


def build_network(x: tf.Tensor, dropout_kp: float = 0.5,
                  is_training: bool = False,
                  reuse: bool = None):
    with tf.variable_scope('embedding_net', 'embedding_net', [x], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.001)):
            x = slim.fully_connected(x, 2048, activation_fn=tf.nn.relu,
                                     scope='fc1')
            x = slim.dropout(x, keep_prob=dropout_kp, is_training=is_training)
            x = slim.fully_connected(x, 2048, activation_fn=tf.nn.relu,
                                     scope='fc2')
            x = slim.dropout(x, keep_prob=dropout_kp, is_training=is_training)
            x = slim.fully_connected(x, 2048, scope='fc3')
    return x
