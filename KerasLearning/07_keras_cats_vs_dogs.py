import os, pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(DIR), 'Data', 'cats_and_dogs_small')
model_path = os.path.join(DIR, 'models')
train_dir = os.path.join(DATA_PATH, 'train')
validation_dir = os.path.join(DATA_PATH, 'validation')

# 图像处理辅助工具模块
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(rescale=1. / 255)  # 验证集不用增强数据

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(150, 150),  # 图像全部调整为150*150
    batch_size=20,
    class_mode='binary'
)


def train():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=100,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=50
    )
    # 四个字段val_loss、val_acc、loss、acc
    with open(os.path.join(model_path, 'history_2.pk'), 'wb') as f:
        pickle.dump(history.history, f)
    model.save(os.path.join(model_path, 'cats_and_dogs_small_2.h5'))


def re_train():
    model = models.load_model(os.path.join(model_path, 'cats_and_dogs_small_2.h5'))
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=100,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=50
    )
    with open(os.path.join(model_path, 'history_2.pk'), 'rb') as f:
        obj = pickle.load(f)
        for key in ['val_loss', 'val_acc', 'loss', 'acc']:
            obj[key] = obj[key] + history.history[key]
        print(obj)
    with open(os.path.join(model_path, 'history_2.pk'), 'wb') as f:
        pickle.dump(obj, f)
    model.save(os.path.join(model_path, 'cats_and_dogs_small_2.h5'))


# train()
for i in range(7):
    re_train()
