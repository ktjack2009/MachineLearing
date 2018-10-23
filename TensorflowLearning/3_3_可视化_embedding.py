'''
Embedding Projector：交互可视化工具
默认情况下，embedding projector 会用 PCA 主成分分析方法将高维数据投影到 3D 空间, 还有一种投影方法是 T-SNE。

三步：
1) Setup a 2D tensor that holds your embedding(s).
    embedding_var = tf.Variable(....)
2) Periodically save your model variables in a checkpoint in LOG_DIR.
    saver = tf.train.Saver()
    saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), step)
3) (Optional) Associate metadata with your embedding.
'''

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from TensorflowLearning.common import deal_label, nn_layer

DIR = os.path.dirname(__file__)
# 训练次数
max_steps = 500
# 测试集中的5000张图片
image_num = 500


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

    # 定义配置文件(前三固定，最后指定metadata文件路径)
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embedding.name
    embed.metadata_path = DIR + '/projector/3_3/metadata.tsv'

    # 原始图片切分(图片可以由原始数据生成)
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
