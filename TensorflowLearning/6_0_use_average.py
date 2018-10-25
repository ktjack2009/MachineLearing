import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
from TensorflowLearning.common import deal_label

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = (train_images / 255.0).reshape([-1, 784]), (test_images / 255.0).reshape([-1, 784])
train_labels, test_labels = deal_label(train_labels), deal_label(test_labels)  # shape = [None, 10]

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_BASE_RATE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减


# 返回的值未经过softmax函数
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        # 不使用滑动平均，直接使用参数当前的取值
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1)
        )
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train():
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1, biases1, weights2, biases2)
    global_step = tf.Variable(0, trainable=False)  # 训练次数参数，不优化

    # 创建指数移动平均的类
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 指定所有需要优化的参数使用滑动平均
    variable_average_op = variable_average.apply(tf.trainable_variables())
    # 使用移动平均模型的预测值
    average_y = inference(x, variable_average, weights1, biases1, weights2, biases2)

    # 高效的计算y的softmax，然后计算与labels的交叉熵，labels接受直接的数字标签
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization

    # 每进行一次完整的训练（60000 // 100 次），学习率下降0.99
    learning_rate = tf.train.exponential_decay(LEARNING_BASE_RATE, global_step, 60000 // 100, LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    # # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    average_2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1)), tf.float32))

    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_feed = {x: test_images, y_: test_labels}

        # 循环的训练神经网络。
        for epoch in range(6):
            for i in range(60000 // BATCH_SIZE):
                xs = train_images[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                ys = train_labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                sess.run(train_op, feed_dict={x: xs, y_: ys})
            acc = sess.run(accuracy, feed_dict=test_feed) * 100
            acc2 = sess.run(average_2, feed_dict=test_feed) * 100
            print(f"After {epoch} training step(s), test accuracy using average model is {acc:.2f}% {acc2:.2f}%")


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
