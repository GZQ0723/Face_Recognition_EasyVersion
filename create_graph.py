import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('face.h5')

plot_model(model,to_file="model.png",show_shapes=True,show_layer_names=True,rankdir='TB')
plt.figure(figsize=(10,10))
img = plt.imread("model.png")
plt.imshow(img)
plt.axis('off')
plt.show()