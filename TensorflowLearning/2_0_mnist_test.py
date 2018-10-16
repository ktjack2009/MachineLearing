import tensorflow as tf
import numpy as np


def base_function(train_images, train_labels, test_images, test_labels):
    n_batch = 60000 // 50

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    # 定义简单的神经网络
    W1 = tf.Variable(tf.zeros([784, 10]))
    b1 = tf.Variable(tf.zeros([10]))
    prediction = tf.nn.softmax(x @ W1 + b1)

    loss = tf.reduce_mean(tf.square(y - prediction))
    train_steps = tf.train.AdamOptimizer(0.01).minimize(loss)

    init = tf.global_variables_initializer()

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(20):
            for n in range(n_batch):
                batch_x, batch_y = train_images[n * 50: n * 50 + 50], train_labels[n * 50: n * 50 + 50]
                sess.run(train_steps, feed_dict={x: batch_x, y: batch_y})
            acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
            acc = acc * 100
            print(f'循环次数={epoch}, 准确率={acc:.2f}%')


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
base_function(train_images, train_labels, test_images, test_labels)
