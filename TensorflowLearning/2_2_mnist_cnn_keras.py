import tensorflow as tf
from Data.mnist_data import train_images, train_labels, test_images, test_labels

train_images = train_images.reshape([-1, 28, 28, 1])
test_images = test_images.reshape([-1, 28, 28, 1])

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
models.fit(train_images, train_labels, batch_size=50, epochs=20)
score = models.evaluate(test_images, test_labels)
print(f'Test score: {score}')
