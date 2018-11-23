from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical


def get_model(flag=0):
    if flag == 0:
        # type 1: Sequential
        model = models.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
        model.add(layers.Dense(10, activation='softmax'))
    else:
        # type 2: api
        input_tensor = layers.Input(shape=(784,))
        x = layers.Dense(32, activation='relu')(input_tensor)
        output_tensor = layers.Dense(10, activation='softmax')(x)
        model = models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((-1, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((-1, 28 * 28)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = get_model()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=128, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)
