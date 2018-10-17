# 显示图结构
# tensorboard --logdir=...

import tensorflow as tf
import numpy as np


def base_method(train_images, train_labels, test_images, test_labels):
    n_batch = 60000 // 50

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)
        lr = tf.Variable(0.001, dtype=tf.float32)

    with tf.name_scope('layer1'):
        with tf.name_scope('W1'):
            W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
        with tf.name_scope('b1'):
            b1 = tf.Variable(tf.zeros([500]) + 0.1)
        with tf.name_scope('L1'):
            L1 = tf.nn.tanh(x @ W1 + b1)
        with tf.name_scope('L1_drop'):
            L1_drop = tf.nn.dropout(L1, keep_prob)

    with tf.name_scope('layer2'):
        with tf.name_scope('W2'):
            W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
        with tf.name_scope('b2'):
            b2 = tf.Variable(tf.zeros([300]) + 0.1)
        with tf.name_scope('L2'):
            L2 = tf.nn.tanh(L1_drop @ W2 + b2)
        with tf.name_scope('L2_drop'):
            L2_drop = tf.nn.dropout(L2, keep_prob)

    with tf.name_scope('layer3'):
        with tf.name_scope('W3'):
            W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
        with tf.name_scope('b3'):
            b3 = tf.Variable(tf.zeros([10]) + 0.1)
        with tf.name_scope('output'):
            prediction = tf.nn.softmax(L2_drop @ W3 + b3)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y - prediction))
    with tf.name_scope('optimizer'):
        train_steps = tf.train.AdamOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as sess:
        sess.run(init)
        tf.summary.FileWriter('./logs/3_0_logs', sess.graph)
        # for epoch in range(15):
        #     sess.run(tf.assign(lr, lr * 0.95))
        #     for n in range(n_batch):
        #         batch_x, batch_y = train_images[n * 50: n * 50 + 50], train_labels[n * 50: n * 50 + 50]
        #         sess.run(train_steps, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        #     acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels, keep_prob: 1})
        #     acc = acc * 100
        #     print(f'循环次数={epoch}, 准确率={acc:.2f}%')


def deal_label(labels):
    _ = np.zeros([len(labels), 10], dtype=np.float32)
    for i in range(len(labels)):
        _[i][labels[i]] = 1
    return _


CLASS = 10
# 训练集有60000条，测试集10000条
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = np.reshape(train_images, [-1, 784])
test_images = np.reshape(test_images, [-1, 784])
train_labels = deal_label(train_labels)
test_labels = deal_label(test_labels)
base_method(train_images, train_labels, test_images, test_labels)
