import os, pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(DIR), 'Data', 'cats_and_dogs_small')
model_path = os.path.join(DIR, 'models')
train_dir = os.path.join(DATA_PATH, 'train')
validation_dir = os.path.join(DATA_PATH, 'validation')

# 图像处理辅助工具模块
train_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)  # 所有图像像素点乘以1./255缩放

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


# 查看图片
# import cv2 as cv
#
# for data, labels in train_generator:
#     for i in range(20):
#         x = data[i]
#         y = labels[i]
#         x = x.astype('uint8')
#         cv.imshow(f'{y}', x)
#         cv.waitKey(0)
#     break
# cv.destroyAllWindows()
# raise KeyboardInterrupt

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
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=100,
        epochs=11,  # 训练11次达到最好效果
        validation_data=validation_generator,
        validation_steps=50
    )
    # 四个字段val_loss、val_acc、loss、acc
    with open(os.path.join(model_path, 'history.pk'), 'wb') as f:
        pickle.dump(history.history, f)
    model.save(os.path.join(model_path, 'cats_and_dogs_small_1.h5'))


train()
