import tensorflow as tf
import os
from PIL import Image
import numpy as np
import shutil

def copy_images_by_class(df, src_dir, dest_dir):
    """
    Copie les images depuis `src_dir` vers `dest_dir`, en les classant par prdtypecode dans des sous-dossiers.

    Args:
        df (pd.DataFrame): DataFrame contenant les noms de fichiers d'images et les codes de produit.
        src_dir (str): Chemin vers le répertoire source contenant les images
        dest_dir (str): Chemin vers le répertoire destination
    """
    for _, row in df.iterrows():
        class_id = str(row['prdtypecode'])

        # Nom du fichier image
        filename = row["filename"]
        src_path = os.path.join(src_dir, filename)
        class_dir = os.path.join(dest_dir, class_id)
        dest_path = os.path.join(class_dir, filename)

        # Créer le dossier de la classe s'il n'existe pas
        os.makedirs(class_dir, exist_ok=True)

        # Copier le fichier s'il existe
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            print(f"Image non trouvée : {src_path}")

def create_balanced_dataset(ds, classes, samples_per_class):
    """
    Crée un dataset équilibré à partir d'un dataset TensorFlow en sur-échantillonnant ou sous-échantillonnant les classes.
    Args:
        ds (tf.data.Dataset): Le dataset d'entrée.
        classes (list): Liste des classes à équilibrer.
        samples_per_class (int): Nombre d'échantillons par classe dans le dataset équilibré.
    Returns:
        tf.data.Dataset: Un dataset équilibré avec le nombre spécifié d'échantillons par classe.
    """ 
    datasets = []
    
    for class_idx in classes:
        # Filtrer le dataset pour cette classe
        class_ds = ds.filter(lambda x, y: tf.equal(y, class_idx))
        num_samples = class_ds.cardinality().numpy()
        # Sur-échantillonner ou sous-échantillonner selon besoin
        if samples_per_class > num_samples:
            # Sur-échantillonnage (répétition)
            class_ds = class_ds.repeat()
            class_ds = class_ds.take(samples_per_class)
        else:
            # Sous-échantillonnage (limitation)
            class_ds = class_ds.take(samples_per_class)
            
        datasets.append(class_ds)
    
    # Combiner tous les datasets de classes
    balanced_ds = datasets[0]
    for ds in datasets[1:]:
        balanced_ds = balanced_ds.concatenate(ds)
    
    # Mélanger le dataset final
    balanced_ds = balanced_ds.shuffle(buffer_size=1000)

    return balanced_ds

def save_balanced_dataset(balanced_ds, output_dir, class_names, format='jpg'):
    """
    Sauvegarde un dataset équilibré d'images dans un répertoire donné, organisé par classes.
    Args:
        balanced_ds (tf.data.Dataset): Le dataset équilibré d'images.
        output_dir (str): Le répertoire de sortie où les images seront sauvegardées.
        class_names (list): Liste des noms de classes pour organiser les images.
        format (str): Format de fichier pour sauvegarder les images (par défaut 'jpg').
    Returns:
        dict: Un dictionnaire contenant le nombre d'images sauvegardées par classe.
    """

    # Créer le répertoire principal s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    # Créer un sous-répertoire pour chaque classe
    for class_name in class_names:  # Adapté à vos 27 classes
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Compteurs pour suivre le nombre d'images par classe
    counts = {i: 0 for i in range(len(class_names))}
    
    # Parcourir le dataset et sauvegarder chaque image
    for image, label in balanced_ds:
        # Convertir en numpy et s'assurer que les valeurs sont dans [0, 255]
        img_array = image.numpy().astype("uint8")

        # Obtenir la classe
        class_idx = int(label.numpy())
        
        # Incrémenter le compteur
        counts[class_idx] += 1
        
        # Définir le chemin de sauvegarde
        img_path = os.path.join(output_dir, class_names[class_idx], f"sample_{counts[class_idx]}.{format}")
        
        # Créer une image PIL et la sauvegarder
        img = Image.fromarray(img_array)
        img.save(img_path)
        
    
    # Afficher les statistiques finales
    print("\nStatistiques de sauvegarde:")
    for class_idx, count in counts.items():
        print(f"Classe {class_idx}: {count} images")
    
    print(f"\nTotal: {sum(counts.values())} images sauvegardées dans {output_dir}")
    
    return counts