import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets


def build_network(x: tf.Tensor,
                  dropout_kp: float = 0.5,
                  is_training: bool = False):
    with tf.variable_scope('embedding_net', 'embedding_net', [x]):
        logits, _ = nets.inception.inception_v3_base(x)
        x = slim.flatten(x)
    return x
