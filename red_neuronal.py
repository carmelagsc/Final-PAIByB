
import tensorflow as tf
import cv2 
import numpy as np
clases = ['Normal', 'Cyst', 'Tumor', 'Stone']

def preprocess(image):
    IMG_SIZE = 128  # Cambia esto al tamaño deseado
    if len(image.shape) == 3 and image.shape[-1] == 3:  # Si la imagen tiene 3 canales (RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    image = tf.expand_dims(image, axis=-1)  # Añade el canal de color (grayscale -> 1 canal)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])  # Redimensiona la imagen
    #image = image / 255.0  # Normaliza los valores a [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)  # Asegúrate de que los valores sean tipo float32
    return image

# Cargar el modelo desde el archivo
modelo = tf.keras.models.load_model('D:/pazar/Documents/Bioingenieria/Quinto/1C/Procesamiento Avanzado de Imágenes/Final/Coronal/my_model_coronal.keras')

imagen=cv2.imread("Cyst-(320).jpg")
imagen=preprocess(imagen)
prediccion = modelo.predict(np.expand_dims(imagen, axis=0))
print("Predicción:", prediccion)
clase_predicha = np.argmax(prediccion, axis=1)
print("Clase predicha:", clase_predicha)

