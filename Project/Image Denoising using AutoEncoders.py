# -*- coding: utf-8 -*-
"""
Spyder Editor

Author : Deep Chokshi
"""
#impoort Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()

#normalize data
X_train = X_train/255
X_test = X_test/255

#add some noise
noise_factor = 0.3
noise_dataset = []
for img in X_train:
    noisy_image = img + noise_factor * np.random.randn(*img.shape)
    noisy_image = np.clip(noisy_image,0,1)
    noise_dataset.append(noisy_image)

noise_dataset = np.array(noise_dataset)

#add noise to testing
noise_factor = 0.1
noise_test_dataset = []
for img in X_test:
    noisy_image = img + noise_factor *np.random.randn(*img.shape)
    noisy_image = np.clip(noisy_image,0,1)
    noise_test_dataset.append(noisy_image)
noise_test_dataset = np.array(noise_test_dataset)

#build and train autoencoder
autoencoder = tf.keras.models.Sequential()
autoencoder.add(tf.keras.layers.Conv2D(filters= 16, kernel_size=3, strides = 2, padding = 'same', input_shape= (28,28,1)))
autoencoder.add(tf.keras.layers.Conv2D(filters= 8, kernel_size=3, strides = 2, padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2D(filters= 8, kernel_size=3, strides = 1, padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters= 16, kernel_size=3, strides = 2, padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters= 1, kernel_size=3, strides = 2, activation = 'sigmoid',padding = 'same'))
autoencoder.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001))
autoencoder.fit(noise_dataset.reshape(-1, 28, 28, 1),X_train.reshape(-1, 28, 28, 1),epochs = 10,batch_size = 200,validation_data = (noise_test_dataset.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1)))
evaluation = autoencoder.evaluate(noise_test_dataset.reshape(-1,28,28,1),X_test.reshape(-1,28,28,1))
predicted = autoencoder.predict(noise_test_dataset[:10].reshape(-1,28,28,1))
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([noise_test_dataset[:10], predicted], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)