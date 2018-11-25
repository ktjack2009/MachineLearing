import os
import cv2 as cv
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(DIR), 'Data', 'cats_and_dogs_small')
model_path = os.path.join(DIR, 'models')
test_dir = os.path.join(DATA_PATH, 'test')
i = 1500
image_1 = os.path.join(test_dir, 'cats', f'cat.{i}.jpg')
image_2 = os.path.join(test_dir, 'dogs', f'dog.{i}.jpg')

image_1 = cv.imread(image_1)
image_2 = cv.imread(image_2)
image_1 = cv.resize(image_1, (150, 150)) / 255.0
image_2 = cv.resize(image_2, (150, 150)) / 255.0
image_1 = np.reshape(image_1, (-1, 150, 150, 3))
image_2 = np.reshape(image_2, (-1, 150, 150, 3))

# cv.imshow('1', image_1)
# cv.imshow('2', image_2)
# cv.waitKey(0)
# cv.destroyAllWindows()

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
features_batch_1 = conv_base.predict(image_1)
features_batch_1 = np.reshape(features_batch_1, (1, -1))
features_batch_2 = conv_base.predict(image_2)
features_batch_2 = np.reshape(features_batch_2, (1, -1))

model = load_model(os.path.join(model_path, 'cats_and_dogs_small_3.h5'))
y1 = model.predict(features_batch_1)
y2 = model.predict(features_batch_2)
print((y1[0], y2[0]))
