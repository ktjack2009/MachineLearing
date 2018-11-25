# 扩展conv_base模型
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(DIR), 'Data', 'cats_and_dogs_small')
model_path = os.path.join(DIR, 'models')
train_dir = os.path.join(DATA_PATH, 'train')
validation_dir = os.path.join(DATA_PATH, 'validation')
test_dir = os.path.join(DATA_PATH, 'test')
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

for layer in conv_base.layers:
    if layer.name.startswith('block5_conv'):
        layer.trainable = True  # 只有block5的参数可训练
    else:
        layer.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
)
validation_generator = validation_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
model.compile(optimizer=optimizers.RMSprop(lr=1e-5), loss='binary_corssentropy', metrics=['acc'])
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=validation_generator, validation_steps=50)
