import tensorflow as tf
from TensorflowLearning.common import deal_label
from tensorflow.keras.callbacks import TensorBoard

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.reshape([-1, 28, 28, 1])
test_images = test_images.reshape([-1, 28, 28, 1])
train_labels = deal_label(train_labels)
test_labels = deal_label(test_labels)

models = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(padding='SAME'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'),
    tf.keras.layers.MaxPooling2D(padding='SAME'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'),
    tf.keras.layers.MaxPooling2D(padding='SAME'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])
models.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy,
               metrics=['accuracy'])
models.fit(train_images, train_labels, batch_size=100, epochs=5, callbacks=[TensorBoard(log_dir='./logs/3_2_logs')])
score = models.evaluate(test_images, test_labels)
print(f'Test score: {score}')
