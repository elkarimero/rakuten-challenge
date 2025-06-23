import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

import numpy as np

def zoom_picture(filepath = None, img_src = None ,threshold=230):
    """
    Nettoie une image en supprimant le fond blanc et en redimensionnant l'image.
    Args:
        filepath (str): Chemin du fichier image à nettoyer.
    Returns:
        numpy.ndarray: L'image nettoyée et redimensionnée.
    """
    if img_src is None:
        if filepath is None:
            raise ValueError("Either 'filepath' or 'img_src' must be provided.")
        # Lire l'image depuis le chemin de fichier
        img_src = cv2.imread(filepath)
    else:
        if not isinstance(img_src, (np.ndarray,)):
            raise ValueError("'img_src' must be a numpy array representing the image.")
        
    # Lire l'image
    image = img_src.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un seuil pour binariser l'image
    # On utilise un seuil de 240 pour détecter les zones très claires (ie pixels blancs)
    # On utilise cv2.THRESH_BINARY_INV pour inverser le seuil et ainsi détecter les pixels blancs du fond
    # Ainsi, les pixels blancs du fond deviennent noirs et les autres pixels deviennent blancs
    # Cela permet de détecter les contours des objets dans l'image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Recherche des contours dans l'image binaire
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 3)

    # Select the biggest bounding box detected
    max_size = 0
    x_max, y_max, w_max, h_max = 0, 0, 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_size = w*h
        if contour_size > max_size: 
            max_size = contour_size
            x_max, y_max, w_max, h_max = x, y, w, h

    # Add margin to bounding box 
    margin = 1 
    image_width, image_height = 500, 500
    x = max(0, x_max - margin)
    w = min(w_max + 2 * margin, image_width - x)
    y = max(0, y_max - margin)
    h = min(h_max + 2 * margin, image_height - y)

    # draw the bounding box on original picture
    rectangle = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # crop picture to eliminate white background
    cropped_image = img_src[y:y+h, x:x+w]

    # find ratio to resize properly
    scale = min(image_width / w, image_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(cropped_image, (new_w, new_h))
    resized = cv2.resize(resized, (224, 224))

    return img_src, gray, binary, contoured, rectangle, resized


# Fonction de prétraitement d'image
def preprocess_image(image: Image.Image):
    # Convertir PIL en array NumPy
    image_np = np.array(image)
    # Convertir RGB (PIL) en BGR (OpenCV)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    _ , _ , _ , _ , _ , zoomed = zoom_picture(img_src=image_cv2)  # Nettoyage de l'image
    resized = tf.image.resize(zoomed, (224, 224))
    
    # Normalisation pour EfficientNetB0 ([-1, 1] si preprocess_input est utilisé)
    image = preprocess_input(resized)

    return tf.expand_dims(image, axis=0), zoomed