import os
import pandas as pd
import numpy as np
from collections import Counter

from PIL import Image, ImageStat
import imagehash
import cv2

def generate_phash(filename, directory_path):
    """
    Génère un hash perceptuel pour une image donnée.
    Args:
        filename (str): Nom du fichier image.
        directory_path (str): Chemin du répertoire contenant l'image.
    Returns:
        str: Le hash perceptuel de l'image.
    """    
    try:
        filepath = os.path.join(directory_path, filename)
        image = Image.open(filepath)
        return str(imagehash.phash(image))
    
    except Exception as e:
        print(f"Erreur lors de l'analyse de {filename}: {str(e)}")
        return np.nan
    
def calculate_mean_var(image):
    """
    Calcule la variance moyenne des canaux de couleur d'une image.
    Args:
        image (numpy.ndarray): L'image pour laquelle calculer la variance.
    Returns:
        float: La variance moyenne des canaux de couleur.
    """
    # Séparer les canaux de couleur
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Calculer la variance de chaque canal
    var_R = np.var(R)
    var_G = np.var(G)
    var_B = np.var(B)

    # Calculer la variance moyenne entre les canaux
    return np.mean([var_R, var_G, var_B])

def calculate_mean_std(image):
    """
    Calcule l'écart type moyen des canaux de couleur d'une image.
    Args:
        image (numpy.ndarray): L'image pour laquelle calculer l'écart type.
    Returns:
        float: L'écart type moyen des canaux de couleur.
    """
    # Séparer les canaux de couleur
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Calculer la variance de chaque canal
    std_R = np.std(R)
    std_G = np.std(G)
    std_B = np.std(B)

    # Calculer la variance moyenne entre les canaux
    return np.mean([std_R, std_G, std_B])

def extract_dominant_color_ratio(image, color_tolerance=10):
    """
    Extrait le ratio de la couleur dominante d'une image.
    Args:
        image (numpy.ndarray): L'image pour laquelle extraire la couleur dominante.
        color_tolerance (int): La tolérance de regroupement des couleurs.
    Returns:
        float: Le ratio de la couleur dominante dans l'image.
    """
    # Transformer l’image en une liste de pixels
    pixels = image.reshape(-1, 3)  # Convertir en une liste de valeurs RGB

    # Convertir en tuples pour comptage
    pixels = [tuple(p) for p in pixels]

    # Regrouper les couleurs proches
    grouped_pixels = []
    for r, g, b in pixels:
        grouped_pixels.append((r // color_tolerance * color_tolerance, 
                               g // color_tolerance * color_tolerance, 
                               b // color_tolerance * color_tolerance))

    # Compter la fréquence de chaque couleur
    color_counts = Counter(grouped_pixels)
    dominant_color = color_counts.most_common()[0]

    # Trouver la proportion de la couleur dominante
    total_pixels = len(pixels)
    dominant_color_count = max(color_counts.values())
    dominant_color_ratio =  (dominant_color_count / total_pixels) * 100

    return dominant_color_ratio
    
def extract_edge_count(image):
    """
    Extrait le nombre de pixels de contours d'une image en utilisant la détection de contours Canny.
    Args:
        image (numpy.ndarray): L'image pour laquelle extraire le nombre de pixels de contours.
    Returns:
        int: Le nombre de pixels de contours détectés dans l'image.
    """
    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Détection des contours avec Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Compter le nombre de pixels de contours
    edge_count = np.sum(edges > 0)

    return edge_count 

def extract_image_features(df, directory_path):
    """ 
    Extrait les caractéristiques des images à partir d'un DataFrame contenant les noms de fichiers.
    Args:
        df (pd.DataFrame): DataFrame contenant les noms de fichiers d'images.
        directory_path (str): Chemin du répertoire contenant les images.
    Returns:
        pd.DataFrame: DataFrame mis à jour avec les caractéristiques des images.
    """
    for index, row in df.iterrows():
        filepath = os.path.join(directory_path, row["filename"])

        img = Image.open(filepath)

        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        df.loc[index, "edge_count"] = extract_edge_count(image_grey) # nombre de pixels de contours
        df.loc[index, "hash"] = generate_phash(row["filename"], directory_path) # hash perceptuel de l'image
        df.loc[index, "mean_var"] = calculate_mean_var(image) # variance moyenne des canaux de couleur
        df.loc[index, "dominant_color_ratio"] = extract_dominant_color_ratio(image) # ratio de la couleur dominante

    return df

def zoom_images(df, orig_dir_path, dest_dir_path):
    """
    Zoom sur les images
    Args:
        df (pd.DataFrame): DataFrame contenant les noms de fichiers d'images.
        orig_dir_path (str): Chemin du répertoire d'origine contenant les images.
        dest_dir_path (str): Chemin du répertoire de destination pour les images zoomées.
    """
    for filename in df["filename"]:
        try:
            filepath = os.path.join(orig_dir_path, filename)
            img = cleanup_picture(filepath, show_images=False)
            copy_filepath = os.path.join(dest_dir_path, filename)
            cv2.imwrite(copy_filepath,img)
            
        except Exception as e:
            print(f"Erreur pour {filename}: {str(e)}")


def cleanup_picture(filepath, show_images = True):
    """
    Nettoie une image en supprimant le fond blanc et en redimensionnant l'image.
    Args:
        filepath (str): Chemin du fichier image à nettoyer.
        show_images (bool): Si True, affiche l'image nettoyée.
    Returns:
        numpy.ndarray: L'image nettoyée et redimensionnée.
    """
    # Lire l'image
    img_src = cv2.imread(filepath)
    image = img_src.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un seuil pour binariser l'image
    # On utilise un seuil de 240 pour détecter les zones très claires (ie pixels blancs)
    # On utilise cv2.THRESH_BINARY_INV pour inverser le seuil et ainsi détecter les pixels blancs du fond
    # Ainsi, les pixels blancs du fond deviennent noirs et les autres pixels deviennent blancs
    # Cela permet de détecter les contours des objets dans l'image
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Recherche des contours dans l'image binaire
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # crop picture to eliminate white background
    cropped_image = img_src[y:y+h, x:x+w]

    # find ratio to resize properly
    scale = min(image_width / w, image_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(cropped_image, (new_w, new_h))

    return resized

def remove_na(data, src_dir_path):
    """
    Supprime les images pouvant être considéré comme des NA du DataFrame
    Args:
        data (pd.DataFrame): DataFrame contenant les noms de fichiers d'images.
        src_dir_path (str): Chemin du répertoire contenant les images.
    Returns:
        pd.DataFrame: DataFrame mis à jour sans les images non disponibles.
    """

    # Placeholder pour les images considérées comme NA
    # Ces images sont considérées comme des NA car elles sont soit vides, soit de mauvaise qualité
    na_pictures = [
        'image_1243417369_product_3773721822.jpg',
        'image_1026011984_product_558486759.jpg', 
        'image_1190250938_product_2832752183.jpg', 
        'image_1268286826_product_3948210877.jpg', 
        'image_1190252023_product_2647272665.jpg', 
        'image_1261394799_product_3898719785.jpg', 
        'image_977145542_product_277513729.jpg', 
        'image_1306155830_product_4164869671.jpg', 
        'image_1100102350_product_1712289008.jpg',
        'image_1142089742_product_884747735.jpg',
        'image_1194474269_product_3160169806.jpg'
        ]
        
        # image_1194474269_product_3160169806.jpg placeholder
    
        # image_1248838417_product_3817897828.jpg
        # image_1248834760_product_3817892731.jpg
        
    
    # Générer un hach perceptuel pour identifier les images similaires dans l'ensemble de données
    na_hash = []
    for filename in na_pictures: 
        filepath = os.path.join(src_dir_path, filename)
        phash = str(imagehash.phash(Image.open(filepath)))
        na_hash.append(phash)

    # Générer un hachage perceptuel pour toutes les images afin d'identifier les na dan le dataset
    pictures_hash = []
    for filename in data["filename"]: 
        pictures_hash.append(generate_phash(filename, src_dir_path))

    # Ajouter le hachage perceptuel au DataFrame
    data["hash"] = pictures_hash

    # Suppression des na
    return data[~data["hash"].isin(na_hash)]



# na ?
# ffff80aad00a80ab 
# e66699e1e65a1895 
# b962df80a20dd4eb
# bb30f0ca0fcfc270
# b86bc3903c8f9666
# d5f5d4926a432cd4
# ead4956a956a952a
# eae8953b953f8540

# 87347a3d6bc96485
# 87347a2f3f8d3085

# e141aeaed134d933