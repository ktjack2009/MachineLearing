import os, pickle

DIR = os.path.dirname(os.path.abspath(__file__))
history_file = os.path.join(DIR, 'models', 'history_2.pk')

with open(history_file, 'rb') as f:
    history = pickle.load(f)
print(len(history['val_loss']))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)  # 1行，2列

axes[0].plot(range(len(history['val_acc'])), history['val_acc'], 'r', label='val_acc')
axes[0].plot(range(len(history['acc'])), history['acc'], 'black', label='acc')
axes[1].plot(range(len(history['val_loss'])), history['val_loss'], 'r', label='val_acc')
axes[1].plot(range(len(history['loss'])), history['loss'], 'black', label='acc')
plt.show()
