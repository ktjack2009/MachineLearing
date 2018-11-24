from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255
x_val = train_images[: 10000]
partial_x_train = train_images[10000:]

train_labels = to_categorical(train_labels)
y_val = train_labels[: 10000]
partial_y_train = train_labels[10000:]

test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255
test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# print(model.summary())  # 显示model结构和详情
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 第5次验证集准确率最高
history = model.fit(x=partial_x_train, y=partial_y_train, epochs=5, batch_size=128, validation_data=(x_val, y_val))
