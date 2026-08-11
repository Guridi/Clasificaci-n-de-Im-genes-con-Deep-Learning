#!/usr/bin/env python3
"""
predict.py - Clasificacion de imagenes externas (Perro / Gato)

Uso:
    python predict.py --image ruta/de/la/imagen.jpg
    python predict.py --image imagenes/perro1.jpg --no-show
    python predict.py                      (pide la ruta de forma interactiva)

Anadido al repositorio base DeepLearning-Classifier para permitir la prediccion
sobre imagenes nuevas (JPG / PNG) que no forman parte del dataset.

NOTA SOBRE EL PREPROCESAMIENTO
------------------------------
El modelo se entreno con `ImageDataGenerator()` SIN el parametro `rescale`, es
decir, con los pixeles en su escala original 0-255 (RGB, redimensionado con la
interpolacion 'nearest' que usa Keras por defecto). Por eso este script escala
las imagenes a ese mismo rango y NO divide entre 255: hacerlo desplazaria la
entrada fuera del dominio visto durante el entrenamiento y degradaria el modelo
hasta el nivel del azar (comprobado: 82.00% -> 50.00% de acierto sobre las 200
imagenes de test). La normalizacion correcta para esta red es, por tanto, la
escala 0-255.
"""

import argparse
import os
import sys

import numpy as np

# Silencia los mensajes informativos de TensorFlow antes de importarlo.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL = os.path.join(
    BASE_DIR, "CatDogTraining-2", "trainedmodels", "vgg16_epoch_13_accuracy_84.55.h5"
)

# Orden alfabetico de las carpetas del dataset (flow_from_directory):
# dataset/train/cat -> 0 , dataset/train/dog -> 1
CLASS_NAMES = ["Gato", "Perro"]

IMG_SIZE = (224, 224)          # tamano de entrada de la red
INTERPOLATION = "nearest"      # el mismo que usa ImageDataGenerator por defecto
PIXEL_SCALE = 255.0            # rango de pixeles visto durante el entrenamiento

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------- #
# Carga y preprocesamiento
# --------------------------------------------------------------------------- #
def load_and_preprocess(img_path):
    """Carga una imagen JPG/PNG y la deja lista para la red neuronal.

    Devuelve (tensor_para_el_modelo, imagen_redimensionada_para_mostrar).
    """
    # load_img convierte a RGB, descartando el canal alfa de los PNG, y
    # redimensiona a 224x224 con la misma interpolacion usada al entrenar.
    pil_img = keras_image.load_img(
        img_path, target_size=IMG_SIZE, color_mode="rgb", interpolation=INTERPOLATION
    )

    # (224, 224, 3) float32 en el rango 0-255
    img_array = keras_image.img_to_array(pil_img)

    # Normalizacion al rango esperado por el modelo (ver nota de cabecera):
    # los valores ya estan en 0-255, se garantiza el rango y el tipo float32.
    img_array = np.clip(img_array, 0.0, PIXEL_SCALE).astype("float32")

    # La red espera un lote: (1, 224, 224, 3)
    batch = np.expand_dims(img_array, axis=0)

    return batch, pil_img


# --------------------------------------------------------------------------- #
# Prediccion
# --------------------------------------------------------------------------- #
def predict(model, batch):
    """Ejecuta el modelo y devuelve (indice, etiqueta, confianza, puntuaciones)."""
    scores = model.predict(batch, verbose=0)[0]     # 2 salidas sigmoid

    index = int(np.argmax(scores))
    label = CLASS_NAMES[index]

    # La ultima capa es 'sigmoid', por lo que las dos salidas son independientes
    # y no suman 1. Se normalizan para expresar la confianza como porcentaje.
    total = float(np.sum(scores))
    confidence = float(scores[index]) / total if total > 0 else 0.0

    return index, label, confidence, scores


def show_result(pil_img, label, confidence, scores, img_path, save_path=None,
                display=True):
    """Muestra la imagen utilizada junto con el resultado de la prediccion."""
    if not display and save_path is None:
        return

    # Sin entorno grafico se usa el backend no interactivo.
    if not display or not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6.6))
    ax.imshow(pil_img)
    ax.axis("off")

    ax.set_title(
        f"Clase predicha: {label}   ({confidence * 100:.2f} % de confianza)",
        fontsize=13, fontweight="bold", pad=12,
    )
    fig.text(
        0.5, 0.045,
        f"Archivo: {os.path.basename(img_path)}\n"
        f"Puntuaciones -> Gato: {scores[0]:.4f}   Perro: {scores[1]:.4f}",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))

    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"Visualizacion guardada en: {save_path}")

    if display and os.environ.get("DISPLAY"):
        plt.show()

    plt.close(fig)


def print_result(img_path, label, confidence, scores):
    """Salida legible por pantalla."""
    print()
    print("=" * 52)
    print(f"  Imagen            : {os.path.basename(img_path)}")
    print(f"  Clase predicha    : {label}")
    print(f"  Confianza         : {confidence * 100:.2f} %")
    print(f"  Puntuacion Gato   : {scores[0]:.4f}")
    print(f"  Puntuacion Perro  : {scores[1]:.4f}")
    print("=" * 52)
    print()


# --------------------------------------------------------------------------- #
# Validacion de la entrada
# --------------------------------------------------------------------------- #
def validate_image_path(img_path):
    """Comprueba que la ruta existe y que el formato es JPG o PNG."""
    if not os.path.exists(img_path):
        return f"la ruta '{img_path}' no existe."
    if not os.path.isfile(img_path):
        return f"'{img_path}' no es un archivo."
    if not img_path.lower().endswith(VALID_EXTENSIONS):
        return (f"formato no admitido en '{os.path.basename(img_path)}'. "
                f"Use JPG o PNG.")
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasifica una imagen externa como Perro o Gato.",
        epilog="Ejemplo: python predict.py --image imagenes/perro1.jpg",
    )
    parser.add_argument("--image", type=str,
                        help="Ruta de la imagen JPG o PNG a clasificar.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Ruta del modelo entrenado (.h5).")
    parser.add_argument("--save", type=str, default=None,
                        help="Guarda la visualizacion en el archivo indicado.")
    parser.add_argument("--no-show", action="store_true",
                        help="No abre la ventana de visualizacion.")
    return parser.parse_args()


def main():
    args = parse_args()

    # El usuario puede indicar la ruta por argumento o de forma interactiva.
    img_path = args.image
    if not img_path:
        try:
            img_path = input("Indique la ruta de la imagen (JPG o PNG): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nOperacion cancelada.")
            return 1
    img_path = os.path.expanduser(img_path.strip().strip('"').strip("'"))

    error = validate_image_path(img_path)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if not os.path.exists(args.model):
        print(f"Error: no se encuentra el modelo '{args.model}'.", file=sys.stderr)
        return 1

    print(f"Cargando modelo: {os.path.relpath(args.model, BASE_DIR)} ...")
    model = tf.keras.models.load_model(args.model, compile=False)

    batch, pil_img = load_and_preprocess(img_path)
    index, label, confidence, scores = predict(model, batch)

    print_result(img_path, label, confidence, scores)
    show_result(pil_img, label, confidence, scores, img_path,
                save_path=args.save, display=not args.no_show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
