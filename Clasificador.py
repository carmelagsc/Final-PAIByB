import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, Sequential, callbacks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, BatchNormalization, LeakyReLU, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import shutil
import random

def clasificador(imagen):
    IMG_SIZE = 128
    imagen_resized = cv2.resize(imagen, (128, 128), interpolation=cv2.INTER_AREA)
    # Crear modelo CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dense(4, activation='softmax')
    ])

    model.load_weights("C:\Users\Equipo\Documents\Final-PAIByB\best_weights_transversal.weights.h5")
    clasificacion=model.predict(imagen_resized)
    print(clasificacion)
    return clasificacion

