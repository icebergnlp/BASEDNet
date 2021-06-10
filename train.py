# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import datetime

# tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x (i.e. if you are in TF2.x, uncomment this)

(train_labels, train_images) = np.load(file="model_data/train_labels.npy", allow_pickle=True), np.load(file="model_data/train_images.npy", allow_pickle=True)
(test_labels, test_images) = np.load(file="model_data/test_labels.npy", allow_pickle=True), np.load(file="model_data/test_images.npy", allow_pickle=True)
(val_labels, val_images) = np.load(file="model_data/val_labels.npy", allow_pickle=True), np.load(file="model_data/val_images.npy", allow_pickle=True)

print("****************************************")
print("TRAIN IMAGES SHAPE:", train_images.shape)
print("TRAIN LABELS SHAPE:", train_labels.size)
print("****************************************")
print("TEST IMAGES SHAPE:", test_images.shape)
print("TEST LABELS SHAPE:", test_labels.shape)
print("****************************************")
print("VAL IMAGES SHAPE:", val_images.shape)
print("VAL LABELS SHAPE:", val_labels.shape)
print("****************************************")

# 1000 * 1000 = 1000000 = 1e6 <- dhSegment's input size

model = keras.Sequential()

# Convolutional layer 1 and maxpool layer 1
model.add(keras.layers.Reshape(target_shape=(1000, 1000, 1), input_shape=(1000, 1000)))
model.add(keras.layers.Conv2D(5, (3,3), activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer 2 and maxpool layer 2
model.add(keras.layers.Conv2D(10, (3,3), activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convoluational layer 3 and global maxpool layer 1
model.add(keras.layers.Conv2D(20, (3,3), activation='relu'))
model.add(keras.layers.GlobalMaxPool2D())

# Fully Connected layer
model.add(keras.layers.Dense(20, activation='relu'))

# Output layer with single neuron which gives 0 for Bad Baselines or 1 for Good Baselines
# Here we use the sigmoid activation function so our output lies within 0 and 1
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Tensorboard
# %tensorboard --logdir logs/fit
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads=True, write_images=True, update_freq='epoch')

# compile the model
model.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=['accuracy'])

# train the model (where the magic happens)
model.fit(x=train_images, y=train_labels, batch_size=1, epochs = 10, shuffle=True, verbose=1, validation_data=(val_images, val_labels))#, callbacks=[tensorboard_callback])

model.save("saved_models/based_cnn_v2.h5")

'''
NOTE: This code creates a visualization of our neural network's model architecture

import visualkeras
from tensorflow import keras
from PIL import ImageFont

font = ImageFont.truetype("UbuntuMono-R.ttf", 30)
model = keras.models.load_model("saved_models/based_cnn_v1.h5")
visualkeras.layered_view(model, legend=True, font=font, spacing=100, index_ignore=[0], scale_xy=0.5, to_file="based_cnnv2.png")
'''