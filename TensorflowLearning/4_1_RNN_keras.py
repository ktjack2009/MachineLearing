import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from common import deal_label

n_inputs = 28  # 一次传1行，传28个像素
max_time = 28  # 一幅图有28行
lstm_size = 100  # 100个cells
n_classes = 10
batch_size = 100

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = (train_images / 255.0).reshape([-1, 28, 28]), (test_images / 255.0).reshape([-1, 28, 28])
train_labels, test_labels = deal_label(train_labels), deal_label(test_labels)  # shape = [None, 10]

model = tf.keras.models.Sequential([
    LSTM(units=lstm_size, batch_input_shape=[None, max_time, n_inputs]),
    Dense(units=n_classes, activation='softmax'),
])
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [TensorBoard(log_dir='./logs/4_1_logs')]
model.fit(train_images, train_labels, batch_size=batch_size, epochs=6, callbacks=callbacks)
score = model.evaluate(test_images, test_labels)
print(score)
