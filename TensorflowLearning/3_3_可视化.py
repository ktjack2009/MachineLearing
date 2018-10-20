import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

DIR = os.path.dirname(__file__)
# 训练次数
max_steps = 500
# 测试集中的5000张图片
image_num = 500


def deal_label(labels):
    _ = np.zeros([len(labels), 10], dtype=np.float32)
    for i in range(len(labels)):
        _[i][labels[i]] = 1
    return _


def weights_variable(shape):
    # 权重初始化
    initial = tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    # 偏置初始化
    initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial)


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


def train():
    # 导入数据
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = (train_images / 255.0).reshape([-1, 784]), (test_images / 255.0).reshape([-1, 784])
    train_labels, test_labels = deal_label(train_labels), deal_label(test_labels)  # shape = [None, 10]

    # 5000张图片
    embedding = tf.Variable(tf.stack(test_images[:image_num]), trainable=False, name='embedding')

    # 输入层
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shape_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shape_input, 10)  # 只展示10张图片

    hidden1 = nn_layer(x, 784, 500, 'layer1')
    prediction = nn_layer(hidden1, 500, 10, 'layer2', act=tf.nn.softmax)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(prediction, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 如果有，删除metadata文件
    if tf.gfile.Exists(DIR + '/projector/3_3/metadata.tsv'):
        tf.gfile.Remove(DIR + '/projector/3_3/metadata.tsv')

    # 将test_labels存入metadata文件中
    with open(DIR + '/projector/3_3/metadata.tsv', 'w') as f:
        f.write("Index\tLabel\n")
        for i in range(image_num):
            f.write("%d\t%d\n" % (i, np.argmax(test_labels[i])))

    # 合并所有的summaries
    merged = tf.summary.merge_all()

    projector_writer = tf.summary.FileWriter(DIR + '/projector/3_3', sess.graph)

    # 定义Saver，保存模型
    saver = tf.train.Saver()

    # 定义配置文件
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embedding.name
    embed.metadata_path = DIR + '/projector/3_3/metadata.tsv'

    # 原始图片切分
    embed.sprite.image_path = DIR + '/projector/data/mnist_10k_sprite.png'
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(projector_writer, config)

    for i in range(max_steps):
        batch_xs, batch_ys = train_images[i * 100: i * 100 + 100], train_labels[i * 100: i * 100 + 100]
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options,
                              run_metadata=run_metadata)
        projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        projector_writer.add_summary(summary, i)
        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
            print("Iter " + str(i) + ", Testing Accuracy= " + str(acc))
    saver.save(sess, DIR + '/projector/3_3/a_model.ckpt', global_step=max_steps)
    projector_writer.close()
    sess.close()


if __name__ == '__main__':
    train()
