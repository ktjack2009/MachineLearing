# 使用CNN实现mnist手写体识别

import tensorflow as tf
from common import deal_label, weights_variable, bias_variable


def conv2d(x, W):
    # 卷积层
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # 池化层
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def base_method(train_images, train_labels, test_images, test_labels):
    n_batch = 60000 // 50

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x_input')
        y = tf.placeholder(tf.float32, [None, 10], name='y_input')

    with tf.name_scope('Conv1'):
        # 初始化第一个卷积层的权重和偏置
        W1 = weights_variable([5, 5, 1, 32], 'W1')  # 32个卷积核从深度为1的平面提取特征
        b1 = bias_variable([32], 'b1')
        conv2d_1 = conv2d(x, W1) + b1
        h_conv1 = tf.nn.relu(conv2d_1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('Conv2'):
        # 初始化第二个卷积层的权重和偏置
        W2 = weights_variable([3, 3, 32, 64], 'W2')
        b2 = bias_variable([64], 'b2')
        conv2d_2 = conv2d(h_pool1, W2) + b2
        h_conv2 = tf.nn.relu(conv2d_2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        # 全连接层1
        flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='flat')
        WC1 = weights_variable([7 * 7 * 64, 1024], name='WC1')
        bc1 = bias_variable([1024], 'bc1')
        h_fc1 = tf.nn.relu(flat @ WC1 + bc1)

    with tf.name_scope('fc2'):
        # 全连接层2
        WC2 = weights_variable([1024, 10], name='WC2')
        bc2 = bias_variable([10], 'bc2')
        prediction = tf.nn.softmax(h_fc1 @ WC2 + bc2)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    train_steps = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(20):
            for n in range(int(n_batch)):
                batch_x, batch_y = train_images[n * 50: n * 50 + 50], train_labels[n * 50: n * 50 + 50]
                sess.run(train_steps, feed_dict={x: batch_x, y: batch_y})
            acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
            acc = acc * 100
            print(f'循环次数={epoch}, 准确率={acc:.2f}%')


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.reshape([-1, 28, 28, 1])
test_images = test_images.reshape([-1, 28, 28, 1])
train_labels = deal_label(train_labels)
test_labels = deal_label(test_labels)
base_method(train_images, train_labels, test_images, test_labels)
