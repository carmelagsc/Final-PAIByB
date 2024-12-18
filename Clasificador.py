import os
import cv2
import numpy as np
import tensorflow as tf 

# Lista de clases (etiquetas)
clases = ['Normal', 'Cyst', 'Tumor', 'Stone']

def detectar_corte(imagen):
    print(f"Tamaño original de la imagen antes del procesamiento: {imagen.shape}")

    if imagen.shape[0] == 512 and imagen.shape[1] == 512:
        print("Tipo de corte detectado: Transversal. Usando pesos para corte transversal.")
        return 'transversal'
    elif imagen.shape[0] != 512 or imagen.shape[1] != 512:
        print("Tipo de corte detectado: Coronal. Usando pesos para corte coronal.")
        return 'coronal'
    else:
        raise ValueError(f"Dimensiones de imagen no reconocidas: {imagen.shape}")

modelo_por_corte = {
    'transversal': "my_model_transversal.keras",
    'coronal': "my_model_coronal.keras",
}

def preprocess(image):
    IMG_SIZE = 128  # Cambia esto al tamaño deseado
    if len(image.shape) == 3 and image.shape[-1] == 3:  # Si la imagen tiene 3 canales (RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    image = tf.expand_dims(image, axis=-1)  # Añade el canal de color (grayscale -> 1 canal)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])  # Redimensiona la imagen
    image = tf.image.convert_image_dtype(image, tf.float32)  # Asegúrate de que los valores sean tipo float32
    return image

def clasificador(imagen):
    tipo_corte = detectar_corte(imagen)
    path_modelo = modelo_por_corte[tipo_corte]

    imagen= preprocess(imagen)
    
    modelo = tf.keras.models.load_model(path_modelo)
    
    prediccion = modelo.predict(np.expand_dims(imagen, axis=0))
    print("Predicción:", prediccion)

    clase_predicha = np.argmax(prediccion, axis=1)
    print("Clase predicha:", clases[clase_predicha[0]])

    # Obtener la probabilidad más alta
    probabilidad_max = np.max(prediccion, axis=-1)[0]


    '''IMG_SIZE = 128
    print(f"Forma de la imagen después de preprocess: {imagen.shape}")
    print(f"Forma de la imagen después de expand_dims: {np.expand_dims(imagen, axis=0).shape}")


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
    model.load_weights(path_pesos)
    
    # Realizar predicción
    clasificacion = model.predict(np.expand_dims(imagen, axis=0))
    
      # Añadir dimensión de batch
    indice_max = np.argmax(clasificacion, axis=-1)[0]
    
    # Obtener la probabilidad más alta
    probabilidad_max = np.max(clasificacion, axis=-1)[0]
    
    # Obtener la clase correspondiente
    clase_predicha = clases[indice_max]
    
    # Mostrar la clase predicha y la probabilidad
    print(f"Clase predicha: {clase_predicha} con probabilidad: {probabilidad_max}")'''
    return clase_predicha, probabilidad_max