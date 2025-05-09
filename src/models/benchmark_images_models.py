import numpy as np
import time
import pandas as pd

# Pour charger les modèles
import tensorflow as tf
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB3, EfficientNetB7,
    ResNet50, ResNet101,
    VGG16, VGG19
)

# Pour importer le datasets
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

# pour serialisation des preprocessing
from keras.saving import register_keras_serializable


# Pour visualiser les performances
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns


# Execution du benchmark
if __name__ == "__main__":
    # Vérification de la version de TensorFlow
    print("TensorFlow version:", tf.__version__)
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    tf.keras.backend.clear_session()
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


    # Chargement du dataset
    #dir_name = "/mnt/c/Users/karim/rakuten/images/data_clean/images_deep/sample"
    dir_name = "/mnt/c/Users/karim/rakuten/images/data_clean/images_deep/sample_balanced"
    img_size = (224, 224)  # Taille cible
    batch_size = 32
    class_names = sorted(os.listdir(dir_name))
    NB_CLASSES = len(class_names)

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

    test_ds = image_dataset_from_directory(
        dir_name,
        image_size=img_size,
        batch_size=batch_size,
        subset="validation",
        validation_split=0.2,
        seed=42
    )