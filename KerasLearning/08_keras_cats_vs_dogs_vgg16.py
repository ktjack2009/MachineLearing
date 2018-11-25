import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# weights：指定模型初始化的权重检查点
# include_top：指定最后是否包含密集连接分类器
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(DIR), 'Data', 'cats_and_dogs_small')
model_path = os.path.join(DIR, 'models')
train_dir = os.path.join(DATA_PATH, 'train')
validation_dir = os.path.join(DATA_PATH, 'validation')
test_dir = os.path.join(DATA_PATH, 'test')

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory=directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
        print(i)
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features, validation_labels = extract_features(validation_dir, 1000)
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))

# test_features, test_labels = extract_features(test_dir, 1000)
from tensorflow.keras import models, layers, optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_features, train_labels, epochs=30, batch_size=20,
                    validation_data=(validation_features, validation_labels))
