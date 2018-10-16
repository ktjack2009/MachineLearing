import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def base_method(x_data, y_data):
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    # 定义神经网络中间层
    W1 = tf.Variable(tf.random_normal([1, 10]))
    b1 = tf.Variable(tf.zeros([1, 10]))
    # L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
    L1 = tf.nn.tanh(x @ W1 + b1)

    # 定义输出层
    W2 = tf.Variable(tf.random_normal([10, 1]))
    b2 = tf.Variable(tf.zeros([1, 1]))
    prediction = tf.nn.tanh(L1 @ W2 + b2)

    # 二次代价函数
    loss = tf.reduce_mean(tf.square(y - prediction))
    # 使用Adam优化
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
    init = tf.global_variables_initializer()

    plt.figure()
    plt.grid()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            sess.run(train_step, feed_dict={x: x_data, y: y_data})
            prediction_value = sess.run(prediction, feed_dict={x: x_data})
            if not i % 20:
                plt.cla()
                plt.scatter(x_data, y_data, c='b', marker='o')
                plt.plot(x_data, prediction_value, 'r-', lw=2)
                plt.pause(0.33)


def keras_method(x_data, y_data):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=10, activation=tf.nn.tanh, name='L1'),
        tf.keras.layers.Dense(units=1, activation=tf.nn.tanh, name='prediction')
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss=tf.keras.losses.mean_squared_error)
    model.fit(x_data, y_data, epochs=100)
    prediction_value = model.predict(x_data)
    plt.scatter(x_data, y_data, color='blue',marker='o', linewidths=1)
    plt.plot(x_data, prediction_value, 'r-', lw=2)
    plt.show()


#  使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise
# base_method(x_data, y_data)
keras_method(x_data, y_data)