import cv2
import numpy as np
import streamlit as st
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import moments_hu
import tempfile

# Función para calcular momentos de Hu y características GLCM
def calcular_textura(imagen_segmentada):
    # Convertir a escala de grises si no lo está
    if len(imagen_segmentada.shape) == 3:
        imagen_segmentada = cv2.cvtColor(imagen_segmentada, cv2.COLOR_BGR2GRAY)
    
    # Calcular GLCM
    glcm = graycomatrix(imagen_segmentada, distances=[1], angles=[0], symmetric=True, normed=True)

    # Calcular las propiedades GLCM
    contraste = graycoprops(glcm, 'contrast')[0, 0]
    homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
    energia = graycoprops(glcm, 'energy')[0, 0]
    entropia = graycoprops(glcm, 'correlation')[0, 0]
    
    # Calcular momentos de Hu
    hu_moment = cv2.HuMoments(cv2.moments(imagen_segmentada)).flatten()
    log_hu_moments = -np.sign(hu_moment) * np.log10(np.abs(hu_moment))

    return contraste, homogeneidad, energia, entropia, log_hu_moments

def region_growing(image, seed, threshold=0):
    # Verificar si la imagen es 3D (color), convertir a escala de grises si es necesario
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows, cols = image.shape
    visited = np.zeros_like(image, dtype=np.uint8)
    region = np.zeros_like(image, dtype=np.uint8)
    stack = [seed]
    seed_value = image[seed[1], seed[0]]

    while stack:
        x, y = stack.pop()
        if visited[y, x] == 0:
            visited[y, x] = 255
            if abs(int(image[y, x]) - int(seed_value)) <= threshold:
                region[y, x] = 255
                if x > 0: stack.append((x-1, y))
                if x < cols-1: stack.append((x+1, y))
                if y > 0: stack.append((x, y-1))
                if y < rows-1: stack.append((x, y+1))

    kernel = np.ones((5, 5), np.uint8)
    region = cv2.dilate(region, kernel, iterations=1)
    return region

def region_growing_from_click(image, recorte, radius=5, threshold=0):
    if image is None:
        return None, None

    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)

    recorte_resized = cv2.resize(recorte, (300, 300), interpolation=cv2.INTER_AREA)

    seed = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            seed.append((x, y))
            print(f"Semilla seleccionada en: ({x}, {y})")
            temp_image = recorte_resized.copy()
            cv2.circle(temp_image, (x, y), radius, (0, 0, 255), 2)
            cv2.imshow("Selecciona la semilla", temp_image)

    cv2.imshow("Selecciona la semilla", recorte_resized)
    cv2.setMouseCallback("Selecciona la semilla", mouse_callback)

    while not seed:
        if cv2.waitKey(1) & 0xFF == 27:
            print("Operación cancelada.")
            cv2.destroyAllWindows()
            return None, None

    cv2.destroyAllWindows()

    segmented_region = region_growing(recorte_resized, seed[0], threshold)

    cv2.imshow("Region Growing", segmented_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    seg_size = cv2.resize(segmented_region, (recorte.shape[1], recorte.shape[0]))
    pixel_seg = np.sum(seg_size != 0)

    return seg_size, pixel_seg

# Función para mostrar el análisis de tumor
def analizar_tumor(imagen_segmentada, recorte):
    # Realizar Region Growing para segmentar el tumor
    segmented_region, pixel_seg = region_growing_from_click(imagen_segmentada, recorte, threshold=40)

    # Multiplicar la segmentación con la imagen original para preservar la textura
    segmentada_con_textura = np.multiply(recorte, segmented_region)

    # Mostrar la imagen segmentada con la textura original
    st.image(segmentada_con_textura, caption="Tumor Segmentado con Textura Original")

    # Calcular las características de textura y momentos de Hu
    contraste, homogeneidad, energia, entropia, hu_moments = calcular_textura(segmentada_con_textura)
    col1, col2 = st.columns(2)
    with col2:
    # Mostrar características de GLCM
        st.subheader("Características GLCM:")
        st.write(f"Contraste: {contraste}")
        st.write(f"Homogeneidad: {homogeneidad}")
        st.write(f"Energía: {energia}")
        st.write(f"Entropía: {entropia}")
    with col1:
        # Mostrar los momentos de Hu
        st.subheader("Momentos de Hu:")
        for i, hu in enumerate(hu_moments):
            st.write(f"Momento de Hu {i + 1}: {hu}")