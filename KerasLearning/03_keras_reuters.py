import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers, optimizers, losses, metrics

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
word_index = reuters.get_word_index()


# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 0, 1, 2是为padding、start of sequence、unknown分别保留的
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(decoded_review)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_train = to_categorical(train_labels)
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

x_test = vectorize_sequences(test_data)
y_test = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])
history = model.fit(x=partial_x_train, y=partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))

# 绘图可知在第9次开始过拟合
# import matplotlib.pyplot as plt
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

result = model.evaluate(x_test, y_test)
print(result)
