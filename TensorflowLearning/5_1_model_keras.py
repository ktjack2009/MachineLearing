import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from TensorflowLearning.common import deal_label

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = (train_images / 255.0).reshape([-1, 784]), (test_images / 255.0).reshape([-1, 784])
train_labels, test_labels = deal_label(train_labels), deal_label(test_labels)  # shape = [None, 10]
batch_size = 50
n_batch = len(train_images) // batch_size


def train_and_save():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, batch_size=50, epochs=5)
    score = model.evaluate(test_images, test_labels)
    print(f'Test score: {score}')
    save_model(model, './models/5_1/my_net.h5')


def load():
    model = load_model('./models/5_1/my_net.h5')
    score = model.evaluate(test_images, test_labels)
    print(f'Test score: {score}')

# train_and_save()
load()
