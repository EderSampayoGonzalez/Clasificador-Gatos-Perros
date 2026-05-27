import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.preprocessing import image as keras_image_processing 
import os
import sys

# --- CAMBIO CLAVE: Usar winsound (solo funciona con archivos .wav) ---
try:
    import winsound
except ImportError:
    winsound = None 
    print("Advertencia: winsound no disponible (solo funciona en Windows). El audio será silenciado.")
# -------------------------------------------------------------------


# --- Archivos ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

MODELO_RUTA = os.path.join(BASE_DIR, "modelo_gatos_perros_otros_final.h5")
IMG_SIZE = (160, 160)
# Tipos de clase: 
CLASS_NAMES = ['Gato', 'Perro', 'Otro'] 

# Rutas de los archivos de audio
DOG_SOUND_PATH = os.path.join(ASSETS_DIR, "dogbark.wav")
CAT_SOUND_PATH = os.path.join(ASSETS_DIR, "catmeow.wav")
ICON_PATH = os.path.join(ASSETS_DIR, "Icon.ico")

# Definir el método de redimensionamiento de PIL
try:
    RESAMPLING_METHOD = Image.Resampling.LANCZOS 
except AttributeError:
    RESAMPLING_METHOD = Image.LANCZOS 

# --- Carga del modelo ---

try:
    tf.get_logger().setLevel('ERROR') 
    model = tf.keras.models.load_model(MODELO_RUTA)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error fatal al cargar el modelo: {e}")
    sys.exit(1) 

# --- Reproduccion de audio ---

def reproducir_sonido(clase):
#       Reproduce el sonido solo si la clase es Perro o Gato.
    
    if winsound:
        sound_path = None
        
        # Reproducir solo si la clase es Perro o Gato
        if clase == 'Perro' and os.path.exists(DOG_SOUND_PATH):
            sound_path = DOG_SOUND_PATH
        elif clase == 'Gato' and os.path.exists(CAT_SOUND_PATH):
            sound_path = CAT_SOUND_PATH
        
        # Si la clase es 'Otro', sound_path sigue siendo None y no se reproduce nada.
        
        if sound_path:
            winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
    else:
        pass


# --- Prediccion ---

def predecir_imagen(img_path):
    """Realiza la predicción multiclase y devuelve la clase ganadora y la confianza."""

    try:
        img = keras_image_processing.load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
        img_array = keras_image_processing.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Silenciar la salida de Keras/TF para las predicciones
        with tf.device('/cpu:0'):
             predictions = model.predict(img_array, verbose=0)
             
        # Lógica Multiclase: Encontrar el índice con la probabilidad más alta
        predicted_class_index = np.argmax(predictions[0])
        resultado = CLASS_NAMES[predicted_class_index]
        confianza = predictions[0][predicted_class_index] * 100
        
        # Opcional: Mostrar todas las probabilidades en la consola
        print("\n===== Probabilidades =====")
        for i, cls in enumerate(CLASS_NAMES):
            print(f"{cls}: {predictions[0][i]*100:.2f}%")
        print(f" Predicción final: {resultado}")
        
        return resultado, confianza

    except Exception as e:
        print(f"Error interno de predicción: {e}")
        return "Error", 0.0

# --- Actualizar UI ---

def cargar_imagen_desde_dialogo():
    ruta_archivo = filedialog.askopenfilename(
        title="Selecciona una Imagen de Perro o Gato",
        filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png")]
    )
    
    if ruta_archivo:
        mostrar_imagen(ruta_archivo)
        clase_predicha, confianza = predecir_imagen(ruta_archivo)
        
        actualizar_etiqueta_prediccion(clase_predicha, confianza)
        
        reproducir_sonido(clase_predicha)


def mostrar_imagen(img_path):
#   Carga y muestra la imagen, redimensionándola para que encaje.
    try:
        # Forzar a RGB al abrir
        img = Image.open(img_path).convert('RGB')
        
        DISPLAY_SIZE_MAX = 350 
        
        img.thumbnail((DISPLAY_SIZE_MAX, DISPLAY_SIZE_MAX), RESAMPLING_METHOD)

        img_tk = ImageTk.PhotoImage(img)

        etiqueta_imagen.config(
            image=img_tk, 
            width=img_tk.width(), 
            height=img_tk.height(),
            text="" 
        )
        etiqueta_imagen.image = img_tk 
        
    except Exception as e:
        etiqueta_imagen.config(text=f"Error al mostrar imagen: {e}", width=400, height=300)
        print(f"Error al mostrar la imagen: {e}")

def actualizar_etiqueta_prediccion(clase, confianza):
    """Actualiza la etiqueta con el resultado y la confianza, usando colores específicos."""
    
    # Lógica de color para 3 clases
    if clase == 'Perro':
        color = "blue"
    elif clase == 'Gato':
        color = "red"
    else: # Clase 'Otro'
        color = "darkgreen" 

    if not clase == 'Perro' and not clase == 'Gato':
        etiqueta_prediccion.config(
            text = f"Predicción: No es un Perro\nni un Gato (Confianza: {confianza:.2f}%)",
            fg=color)
    else:
        etiqueta_prediccion.config(
            text=f"Predicción: Es un {clase.upper()}\n(Confianza: {confianza:.2f}%)", 
            fg=color
        )

# --- Configuracion del UI ---

root = tk.Tk()
root.title("Clasificador de Imagenes (Perro/Gato/Otro)")
root.geometry("450x550")
root.resizable(False, False)

# --- Cambio de icono ---
if os.path.exists(ICON_PATH):
    try:
        root.iconbitmap(ICON_PATH)
    except Exception:
        try:
            img = Image.open(ICON_PATH)
            img_tk = ImageTk.PhotoImage(img)
            root.iconphoto(True, img_tk)
        except Exception as e:
            pass


etiqueta_titulo = tk.Label(root, text="Clasificador de Perros y Gatos", font=("Arial", 16, "bold"), pady=10)
etiqueta_titulo.pack()


marco_imagen = tk.Frame(root, relief=tk.SUNKEN, bd=2, padx=5, pady=5)
marco_imagen.pack(pady=10)
boton_cargar = tk.Button(root, text="Cargar Imagen", command=cargar_imagen_desde_dialogo,
                        font=("Arial", 12), bg="#4CAF50", fg="white", relief=tk.RAISED, bd=3)
boton_cargar.pack(pady=10, padx=20, fill=tk.X)

etiqueta_imagen = tk.Label(marco_imagen, text="[Presiona 'Cargar Imagen' para empezar]", width=40, height=15, bg="#eeeeee")
etiqueta_imagen.pack()

etiqueta_prediccion = tk.Label(root, text="Predicción: N/A", font=("Arial", 14, "bold"),
                           bg="#f0f0f0", pady=15, relief=tk.RIDGE, bd=2)
etiqueta_prediccion.pack(pady=10, padx=20, fill=tk.X)

root.mainloop()
