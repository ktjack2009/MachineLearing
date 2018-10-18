import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from PIL import Image


def deal_label(labels):
    _ = np.zeros([len(labels), 10], dtype=np.float32)
    for i in range(len(labels)):
        _[i][labels[i]] = 1
    return _


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.sqrt(var - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.reshape([-1, 784])
test_images = test_images.reshape([-1, 784])
train_labels = deal_label(train_labels)
test_labels = deal_label(test_labels)

# 运行次数
max_steps = 3001
# 图片数量
image_num = 5000

embedding = tf.Variable(tf.stack(test_images[:image_num]), trainable=False, name='embedding')
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

with tf.name_scope('image'):
    image_shape_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shape_input, 10)    # 只方10张

# 显示图片
# im = train_images[0].reshape(28,28) * 255
# im = Image.fromarray(im)
# im.show()
