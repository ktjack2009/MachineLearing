'''
LeNet5 model
    第0层，输入层：接受输入32*32*1
    第1层，卷积层：过滤器5*5，深度为6，不使用全0填充，输出尺寸32-5+1=28，下一层节点矩阵28*28*6
    第2层，池化层：输出矩阵14*14*6
    第3层，卷积层：过滤器5*5，深度为16，不使用全0填充，下一层节点矩阵10*10*16
    第4层，池化层：输出矩阵5*5*16
    第5层，全连接层：输出节点120个
    第6层，全连接层：输出节点84个
    第7层，全连接层：输出节点10个
'''

import tensorflow as tf
from TensorflowLearning.common import deal_label

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = (train_images / 255.0).reshape([-1, 28, 28, 1]), (test_images / 255.0).reshape(
    [-1, 28, 28, 1])
train_labels, test_labels = deal_label(train_labels), deal_label(test_labels)  # shape = [None, 10]

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(padding='valid'),  # 14 * 14 * 6
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu'),  # 10 * 10 * 16
    tf.keras.layers.MaxPool2D(padding='valid'),  # 5 * 5 * 16
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
opt = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=100, epochs=5)
score = model.evaluate(test_images, test_labels)
print('score: {}'.format(score))
