import tensorflow as tf
from Data.mnist_data import train_images, train_labels, test_images, test_labels


def base_method(train_images, train_labels, test_images, test_labels):
    n_batch = 60000 // 50

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    lr = tf.Variable(0.001, dtype=tf.float32)

    W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
    b1 = tf.Variable(tf.zeros([500]) + 0.1)
    L1 = tf.nn.tanh(x @ W1 + b1)
    L1_drop = tf.nn.dropout(L1, keep_prob)

    W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
    b2 = tf.Variable(tf.zeros([300]) + 0.1)
    L2 = tf.nn.tanh(L1_drop @ W2 + b2)
    L2_drop = tf.nn.dropout(L2, keep_prob)

    W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]) + 0.1)
    prediction = tf.nn.softmax(L2_drop @ W3 + b3)

    loss = tf.reduce_mean(tf.square(y - prediction))
    train_steps = tf.train.AdamOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(20):
            sess.run(tf.assign(lr, lr * 0.95))
            for n in range(n_batch):
                batch_x, batch_y = train_images[n * 50: n * 50 + 50], train_labels[n * 50: n * 50 + 50]
                sess.run(train_steps, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels, keep_prob: 1})
            acc = acc * 100
            print(f'循环次数={epoch}, 准确率={acc:.2f}%')


CLASS = 10
# 训练集有60000条，测试集10000条
base_method(train_images, train_labels, test_images, test_labels)
