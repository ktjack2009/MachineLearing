import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(DIR), 'Data', 'cats_and_dogs_small')
model_path = os.path.join(DIR, 'models', 'cats_and_dogs_small_2.h5')
image_path = os.path.join(DATA_PATH, 'test', 'cats', 'cat.1700.jpg')

img = image.load_img(image_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

model = load_model(model_path)
layer_outputs = [layer.output for layer in model.layers[:8]]  # 提取前8层的输出
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)  # 创建一个模型，给定模型输出
activations = activation_model.predict(img_tensor)
layer_activation = activations[0]
for i in range(layer_activation.shape[3]):
    plt.matshow(layer_activation[0, :, :, i], cmap='viridis')
plt.show()
