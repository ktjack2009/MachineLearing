import tensorflow as tf
import numpy as np


def deal_label(labels):
    _ = np.zeros([len(labels), 10], dtype=np.float32)
    for i in range(len(labels)):
        _[i][labels[i]] = 1
    return _


def variable_summary(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值散点图
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差散点图
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)  # 直方图


def base_method(train_images, train_labels, test_images, test_labels):
    n_batch = 60000 // 50

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            W = tf.Variable(tf.truncated_normal([784, 10], mean=0, stddev=0.1), name='W')
            variable_summary(W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.constant(0.1, tf.float32, shape=[10]), name='b')
            variable_summary(b)
        with tf.name_scope('prediction'):
            prediction = tf.nn.softmax(x @ W + b)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y - prediction))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('Optimizer'):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # 合并所有的summary
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('./logs/3_1_logs', sess.graph)
        for epoch in range(50):
            for n in range(n_batch):
                batch_x, batch_y = train_images[n * 50: n * 50 + 50], train_labels[n * 50: n * 50 + 50]
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch_x, y: batch_y})
            acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
            acc = acc * 100
            print(f'循环次数={epoch}, 准确率={acc:.2f}%')
            writer.add_summary(summary, epoch)


# 训练集有60000条，测试集10000条
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = np.reshape(train_images, [-1, 784])
test_images = np.reshape(test_images, [-1, 784])
train_labels = deal_label(train_labels)
test_labels = deal_label(test_labels)
base_method(train_images, train_labels, test_images, test_labels)
