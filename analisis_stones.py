import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt




def clusters(recorte):
    if isinstance(recorte, np.ndarray):
        if recorte.size == 0:
            print("El recorte está vacío.")
            return None  # O alguna acción alternativa para manejar este caso 
        pixels = recorte.reshape((-1, 1))  # Aplana la imagen en un arreglo de píxeles

        # Aplicar K-means (n_clusters=2 para dos clases, por ejemplo)
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(pixels)

        # Obtener las etiquetas (clusters) y la imagen segmentada
        labels = kmeans.labels_  # Etiquetas de los píxeles

        # Reconvertir las etiquetas a la forma original de la imagen
        segmented_image = labels.reshape(recorte.shape[0], recorte.shape[1])
    
        return segmented_image
    else:
        print("Recorte no es un arreglo de NumPy.")
        return None 


def region_growing(image, seed, threshold=0):
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



def region_growing_from_click(image,recorte,radius=5, threshold=0):
    # Verificar y convertir a uint8 si es necesario
    if image is None:
        return None, None

    if image.all()== None:
        return None, None

    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)

    # Redimensionar la imagen para visualización
    image_resized = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)


    seed = []

    def mouse_callback(event, x, y, flags, param):
        """
        Callback para capturar el clic y seleccionar la semilla.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Guardar la semilla seleccionada
            seed.append((x, y))
            print(f"Semilla seleccionada en: ({x}, {y})")
            # Dibujar un círculo en la imagen redimensionada
            temp_image = image_resized.copy()
            cv2.circle(temp_image, (x, y), radius, (0, 0, 255), 2)  # Círculo rojo
            cv2.imshow("Selecciona la semilla", temp_image)

    # Mostrar la imagen para selección
    cv2.imshow("Selecciona la semilla", cv2.resize(recorte, (300, 300), interpolation=cv2.INTER_AREA))
    cv2.setMouseCallback("Selecciona la semilla", mouse_callback)

    print("Haz clic en el centro del círculo para seleccionar la semilla.")
    while not seed:
        if cv2.waitKey(1) & 0xFF == 27:  # Presionar ESC para cancelar
            print("Operación cancelada.")
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

    # Ejecutar Region Growing con la semilla seleccionada
    segmented_region = region_growing(image_resized, seed[0], threshold)

    # Mostrar la región segmentada
    cv2.imshow("Region Growing", segmented_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    seg_size=cv2.resize(segmented_region,  (recorte.shape[1],recorte.shape[0]) )
    pixel_seg = np.sum(seg_size!= 0)

    return segmented_region , pixel_seg


def tamaño_corte(obj, x=0.3, y=0.3): #x e y son los tamaños del pixel
    obj_area=obj*x*y #mm^2
    return obj_area


def proba_piedra(tam):
    if tam < 2:
        return "85% de probabilidad que pase naturalmente"
    elif 2 <= tam <= 4:
        return "80% de probabilidad que pase naturalmente"
    elif tam == 4:
        return "80% de probabilidad que pase naturalmente"
    elif 4 < tam <= 7:
        return "60% de probabilidad que pase naturalmente"
    elif 7 < tam < 10:  # Entre 7 mm y 1 cm
        return "30% de probabilidad que pase naturalmente"
    elif 10 <= tam <= 20:  # Entre 1 cm y 2 cm
        return "Intervención quirurgica"
    elif tam > 20:  # Mayor a 2 cm
        return "Intervención quirurgica"
    else:
        return "Invalid measurement"