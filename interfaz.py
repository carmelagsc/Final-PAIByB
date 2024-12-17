import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import analisis_stones

st.title('Final de PAIByB')

# Inicializar session_state para la imagen y selección si no existen
if 'imagen' not in st.session_state:
    st.session_state.imagen = None
if 'seleccion' not in st.session_state:
    st.session_state.seleccion = None

# Función para usar OpenCV y capturar una selección rectangular
def seleccionar_area_cv2(image_path):
    # Cargar imagen con OpenCV
    image = cv2.imread(image_path)
    roi = cv2.selectROI("Seleccionar Área", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return roi  # Devuelve coordenadas (x, y, ancho, alto)

# Menú lateral
st.sidebar.title("Menú")
opcion = st.sidebar.radio("Navegación", ["Cargar Imagen CT", "Análisis"])

# Lógica para cargar imagen
if opcion == "Cargar Imagen CT":
    st.header("Cargar Imagen CT")
    label = " *Cargar imagen CT* "
    imagen = st.file_uploader(label, type=["png", "jpg", "jpeg"])
    
    # Guardar la imagen en session_state si se carga
    if imagen is not None:
        st.session_state.imagen = imagen
        st.image(imagen, caption="Imagen cargada")
        st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{imagen.getvalue().hex()}" 
                 style="max-width: 100%; height: auto;"/>
        </div>
        """, 
        unsafe_allow_html=True
    )
    else:
        if st.session_state.imagen is not None:
            st.image(st.session_state.imagen, caption="Imagen cargada desde la sesión")
        else:
            st.write("Por favor, carga una imagen para visualizarla.")

# Lógica de Análisis con OpenCV
elif opcion == "Análisis":
    st.header("Análisis")
    if st.session_state.imagen is not None:
        # Mostrar la imagen cargada en la sección de Análisis
        st.subheader("Imagen cargada:")
        st.image(st.session_state.imagen, caption="Imagen para análisis")
        
        # Guardar imagen temporalmente para OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tfile.write(st.session_state.imagen.read())

        # Botón para realizar la selección
        st.write("Selecciona un área en la imagen ")
        if st.button("Realizar selección"):
            # Capturar selección usando OpenCV
            roi = seleccionar_area_cv2(tfile.name)

            # Guardar la selección en session_state
            if roi != (0, 0, 0, 0):  # Verificar que se seleccionó algo
                x, y, w, h = roi
                st.session_state.seleccion = (x, y, w, h)
                st.success(f"Área seleccionada: x={x}, y={y}, ancho={w}, alto={h}")

                # Mostrar la imagen recortada
                image = cv2.imread(tfile.name)
                recorte = image[y:y+h, x:x+w]
                recorte= recorte[:, :, 0] 
                st.image(recorte, caption="Área seleccionada")
                segmentacion=analisis_stones.clusters(recorte)
                riñon, px_riñon= analisis_stones.region_growing_from_click(segmentacion, recorte)
                tam_r=analisis_stones.tamaño_corte(px_riñon)
                piedra, px_piedra= analisis_stones.region_growing_from_click(segmentacion, recorte)
                tam_c=analisis_stones.tamaño_corte(px_piedra)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(riñon, caption="Riñon segmentado")
                    st.text(f'El tamaño del corte del riñon es: {tam_r} mm2')

                with col2:
                    st.image(piedra,caption="Piedra segmentada")
                    st.text(f'El tamaño del corte del coágulo es: {tam_c} mm2' )
            else:
                st.warning("No se realizó ninguna selección.")
        

        # Mostrar selección previa si existe
        if st.session_state.seleccion:
            st.write(f"Selección guardada: {st.session_state.seleccion}")
            recorte=st.session_state.seleccion
            segmentacion=analisis_stones.clusters(recorte)
            riñon, px_riñon= analisis_stones.region_growing_from_click(segmentacion, recorte)
            piedra, px_piedra= analisis_stones.region_growing_from_click(segmentacion, recorte)

    else:
        st.warning("Por favor, carga una imagen primero desde la sección 'Cargar Imagen CT'.")
