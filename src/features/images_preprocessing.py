from image_features import extract_image_features, zoom_images, remove_na

# pour le resampling des images
from images_dataset_resampling import create_balanced_dataset, save_balanced_dataset, copy_images_by_class
from tensorflow.keras.preprocessing import image_dataset_from_directory

import pandas as pd
import time
import numpy as np

def main():

    # load raw train dataset
    X_train = pd.read_csv("../../data/raw/X_train.csv", sep=",",index_col=0)
    Y_train = pd.read_csv("../../data/raw/Y_train.csv", sep=",",index_col=0)
    X_train["filename"] = "image_" + X_train["imageid"].astype(str) + "_product_" + X_train["productid"].astype(str) + ".jpg" # add filename for later processing
    train_pictures = pd.concat([X_train[["filename"]], Y_train], axis=1)


    img_train_rep = "/mnt/c/Users/karim/rakuten/images/data_raw/image_train"
    img_train_cleaned_rep = "/mnt/c/Users/karim/rakuten/images/data_clean/image_train"
    img_tmp = "/mnt/c/Users/karim/rakuten/images/tmp"
    img_train_cleaned_resampled_rep = "/mnt/c/Users/karim/rakuten/images/data_clean/image_train_resampled"

    ################
    # Zoom sur l'objet
    ################

    print("start: zoom images")
    start = time.time()

    # zoom sur les images pour se concentrer sur l'objet d'intérêt
    zoom_images(train_pictures, img_train_rep, img_train_cleaned_rep)

    end = time.time()
    print("temps d'exécution:",end-start)

    ################
    # Extraction des features utiles
    ################

    print("start: Extract image features")
    start = time.time()

    # extraction des features utiles des images
    # (nombre de pixels de contours, hash perceptuel, variance moyenne des canaux de couleur, ratio de la couleur dominante)
    train_pictures = extract_image_features(train_pictures, img_train_cleaned_rep)
    train_pictures.to_csv("../../data/interim/train_pictures.csv")

    end = time.time()
    print("temps d'exécution:",end-start) 

    ################
    # Clean up
    ################

    print("start: Clean up")
    start = time.time()

    # suppression des images identifiées comme des valeurs manquantes
    train_pictures = remove_na(train_pictures, img_train_rep) 

    # suppresion des doublons          
    train_pictures = train_pictures.drop_duplicates(subset=["hash"])

    end = time.time()
    print("temps d'exécution:",end-start)

    ##################
    # Resampling images
    ##################

    print("start: Resampling images")
    start = time.time()

    # Création d'un répertoire temporaire pour les images qui respecte la structure attendue par image_dataset_from_directory
    copy_images_by_class(train_pictures, img_train_cleaned_rep, img_tmp)

    img_size = (224, 224)  # Taille cible optimisée pour les modèles de vision par ordinateur

    # chargement du dataset d'images
    ds = image_dataset_from_directory(
        img_tmp,
        image_size=img_size,
        batch_size=128,
        seed=42
    )

    # Compter les occurrences de chaque classe
    class_labels = []
    for _, labels in ds:
        class_labels.extend(labels.numpy().tolist())
    
    classes, counts = np.unique(class_labels, return_counts=True)

    # Défini la médiane comme seuil pour identifier les classes minoritaires et majoritaires
    target_samples = threshold = np.median(counts)  
    # Créer un dataset équilibré de target_samples images par classe
    balanced_train_ds = create_balanced_dataset(ds.unbatch(), classes, target_samples)
    # Sauvegarder le dataset équilibré dans un répertoire organisé par classes
    save_balanced_dataset(balanced_train_ds,img_train_cleaned_resampled_rep, ds.class_names )
    end = time.time()
    print("temps d'exécution:",end-start)


    # Save processed features
    train_pictures.to_csv("../../data/processed/train_pictures.csv")
    train_pictures[["filename", "hash"]].to_csv("../../data/processed/X_train_pictures.csv")
    train_pictures[["prdtypecode"]].to_csv("../../data/processed/Y_train_pictures.csv")

if __name__ == "__main__":
    main()