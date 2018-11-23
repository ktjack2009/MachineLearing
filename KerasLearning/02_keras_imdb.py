import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models, layers, optimizers, losses, metrics

# num_words=10000，保留训练数据中前10000个最常出现的单词，舍弃低频单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# word_index = imdb.get_word_index()  # 将单词映射为整数的索引
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 0, 1, 2是为padding、start of sequence、unknown分别保留的
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(decoded_review)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# 准备数据，留出验证集
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
x_val = x_train[: 10000]
partial_x_train = x_train[10000:]

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
y_val = y_train[: 10000]
partial_y_train = y_train[10000:]

# 构造模型
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# model.fit()返回History对象，对象有成员history，包含训练中的所有数据
# 通过绘制训练损失和验证损失，发现epochs=4时验证损失最小，泛化能力最强
history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)
print(results)
