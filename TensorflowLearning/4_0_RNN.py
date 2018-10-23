import tensorflow as tf
import numpy as np
from TensorflowLearning.common import deal_label, weights_variable, bias_variable

n_inputs = 28  # 一次传1行，传28个像素
max_time = 28  # 一幅图有28行
lstm_size = 100  # 100个cells
n_classes = 10
batch_size = 50
n_batch = 60000 / 50


# 构造模型
def train(train_images, train_labels, test_images, test_labels):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    x_image = tf.reshape(x, [-1, max_time, n_inputs])

    W = weights_variable([lstm_size, n_classes])
    b = bias_variable([n_classes])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_image, dtype=tf.float32)
    prediction = tf.nn.softmax(final_state[1] @ W + b)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(prediction, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        tf.summary.FileWriter('./logs/3_4_logs', sess.graph)
        for epoch in range(6):
            for i in range(int(n_batch)):
                batch_xs = train_images[i * batch_size: (i + 1) * batch_size]
                batch_ys = train_labels[i * batch_size: (i + 1) * batch_size]
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
            acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
            print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = np.reshape(train_images, [-1, 784])
test_images = np.reshape(test_images, [-1, 784])
train_labels = deal_label(train_labels)
test_labels = deal_label(test_labels)
train(train_images, train_labels, test_images, test_labels)
