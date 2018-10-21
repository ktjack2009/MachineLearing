import tensorflow as tf
from tensorflow.keras.utils import to_categorical


def deal_label(labels, n_classes=10):
    return to_categorical(labels, n_classes)


def weights_variable(shape, name=None):
    # 权重初始化
    initial = tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    # 偏置初始化
    initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, name=name)


def variable_summaries(var):
    # 参数概要
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.sqrt(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 神经网络层
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W = weights_variable([input_dim, output_dim])
            variable_summaries(W)
        with tf.name_scope('bias'):
            b = bias_variable([output_dim])
            variable_summaries(b)
        with tf.name_scope('Wx_plus_b'):
            pre_activate = input_tensor @ W + b
            tf.summary.histogram('pre_activate', pre_activate)
        activations = act(pre_activate, name='activation')
        tf.summary.histogram('activation', activations)
        return activations
