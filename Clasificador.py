import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 

# Lista de clases (etiquetas)
clases = ['Normal', 'Cyst', 'Tumor', 'Stone']

def clasificador(imagen):
    # Crear modelo CNN con input_shape definido
    imagen = preprocess(imagen)
    IMG_SIZE = 128
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Cargar pesos preentrenados
    model.load_weights("C:/Users/Equipo/Documents/Final-PAIByB/best_weights_transversal.weights.h5")
    
    # Realizar predicción
    clasificacion = model.predict(np.expand_dims(imagen, axis=0))
      # Añadir dimensión de batch
    indice_max = np.argmax(clasificacion, axis=-1)[0]
    
    # Obtener la probabilidad más alta
    probabilidad_max = np.max(clasificacion, axis=-1)[0]
    
    # Obtener la clase correspondiente
    clase_predicha = clases[indice_max]
    
    # Mostrar la clase predicha y la probabilidad
    print(f"Clase predicha: {clase_predicha} con probabilidad: {probabilidad_max}")
    return clase_predicha, probabilidad_max


def preprocess(image):
    IMG_SIZE = 128  # Cambia esto al tamaño deseado
    if len(image.shape) == 3 and image.shape[-1] == 3:  # Si la imagen tiene 3 canales (RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    image = tf.expand_dims(image, axis=-1)  # Añade el canal de color (grayscale -> 1 canal)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])  # Redimensiona la imagen
    image = tf.image.convert_image_dtype(image, tf.float32)  # Asegúrate de que los valores sean tipo float32
    return image



