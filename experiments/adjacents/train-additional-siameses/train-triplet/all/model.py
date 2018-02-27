import tensorflow as tf
import tensorflow.contrib.slim.nets as nets


def build_network(x: tf.Tensor,
                  dropout_kp: float = 0.5,
                  is_training: bool = False,
                  reuse: bool = None):
    x, endpoints = nets.alexnet.alexnet_v2(x,
                                           is_training=is_training,
                                           dropout_keep_prob=dropout_kp)
    return x
