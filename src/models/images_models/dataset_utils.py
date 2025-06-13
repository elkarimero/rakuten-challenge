# utils.py
import os

# Pour importer le datasets
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def build_dataset_from_directory(dir_name, img_size=(224, 224), batch_size=64):
    """
    Charge un dataset d'images à partir d'un répertoire donné.
    
    Args:
        dir_name: Chemin du répertoire contenant les images organisées en sous-dossiers par classe.
        img_size: Taille des images après redimensionnement.
        batch_size: Nombre d'images par lot.
    
    Return:
        Un tuple contenant les datasets d'entraînement et de validation.
    """

    train_ds = image_dataset_from_directory(
        dir_name,
        image_size=img_size,
        batch_size=batch_size,
        subset="training",
        validation_split=0.2,
        seed=42
    )

    val_ds = image_dataset_from_directory(
        dir_name,
        image_size=img_size,
        batch_size=batch_size,
        subset="validation",
        validation_split=0.2,
        seed=42
    )
    
    return train_ds, val_ds
    
def preprocess_ds(image, label, preprocess_fn=None, augmentation_fn=None):
    """
    Prétraite une image et son label en appliquant des fonctions de prétraitement et d'augmentation.
    Args:
        image: Image à prétraiter.
        label: Label associé à l'image.
        preprocess_fn: Fonction de prétraitement à appliquer à l'image.
        augmentation_fn: Fonction d'augmentation à appliquer à l'image.
    Returns:
        Tuple contenant l'image prétraitée et son label.
    """
    if augmentation_fn and callable(augmentation_fn):
        image = augmentation_fn(image)
    if preprocess_fn and callable(preprocess_fn):
        image = preprocess_fn(image)
    return image, label