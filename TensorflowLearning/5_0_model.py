import tensorflow as tf
from TensorflowLearning.common import deal_label, nn_layer

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = (train_images / 255.0).reshape([-1, 784]), (test_images / 255.0).reshape([-1, 784])
train_labels, test_labels = deal_label(train_labels), deal_label(test_labels)  # shape = [None, 10]
batch_size = 50
n_batch = len(train_images) // batch_size


def train(train_images, train_labels, test_images, test_labels):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    prediction = nn_layer(x, 784, 10, 'L1', act=tf.nn.softmax)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    train_steps = tf.train.AdamOptimizer(0.001).minimize(loss)

    correct = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    def save_model(path):
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(5):
                for i in range(n_batch):
                    batch_xs = train_images[i * batch_size: (i + 1) * batch_size]
                    batch_ys = train_labels[i * batch_size: (i + 1) * batch_size]
                    sess.run(train_steps, feed_dict={x: batch_xs, y: batch_ys})
                acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
                print(f'训练{epoch}, 准确率{acc}')
            saver.save(sess, path)
        print('over')

    # save_model('./models/5_0/my_net.ckpt')

    def load_model(path):
        with tf.Session() as sess:
            saver.restore(sess, path)
            acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
            print(f'准确率{acc}')
        print('over')

    load_model('./models/5_0/my_net.ckpt')


train(train_images, train_labels, test_images, test_labels)
