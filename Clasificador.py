import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 

#from tensorflow.keras.Sequential import layers, models, regularizers, Sequential, callbacks
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, BatchNormalization, LeakyReLU, Input, Activation


def clasificador(imagen):
    
    imagen=imagen[:,:,0]
    print("AAAAAAAAAAAAAA", imagen.shape)
    IMG_SIZE = 128
    imagen_resized = cv2.resize(imagen, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    if len(imagen_resized.shape) == 2:  # Si la imagen es 2D (height, width)
        imagen_resized = tf.expand_dims(imagen_resized, axis=-1)  # Añade el canal de color
    print("AAAAAAAAAAAAAA", imagen_resized.shape)
    # Crear modelo CNN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(4, activation='softmax'),
    ])

    model.load_weights("C:/Users/Equipo/Documents/Final-PAIByB/best_weights_transversal.weights.h5")
    clasificacion=model.predict(imagen_resized)
    print(clasificacion)
    return clasificacion

