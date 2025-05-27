import pandas as pd
import numpy as np
import os

# librairies de visualisation
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":

     # Définir le chemin du dossier où se trouve ce script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Chemin du dossier de données brutes
    raw_data_dir_path = os.path.normpath(os.path.join(BASE_DIR, '../../data/raw'))
    # Chemin du dossier de sauvegarde
    save_dir_path = os.path.normpath(os.path.join(BASE_DIR, '../../reports/analyse_explo_images'))

    X_train = pd.read_csv(os.path.join(raw_data_dir_path, 'X_train.csv'), sep=",",index_col=0)
    y_train = pd.read_csv(os.path.join(raw_data_dir_path, 'Y_train.csv'), sep=",",index_col=0)

    image_train = pd.read_csv(os.path.join(raw_data_dir_path, 'image_train.csv'), sep=",",index_col=0)
    
    # Enregistre le describe dans un CSV pour avoir un aperçu des données
    desc = image_train.describe()
    desc.to_csv(os.path.join(save_dir_path, 'image_train_describe.csv'))  
    print("le dataset contient {} doublons".format(image_train.duplicated(subset="hash").sum()))

    # merge des dataframe pour faciliter l'exploration
    train = pd.concat([X_train, y_train], axis=1)
    train = pd.merge(train, image_train, how="inner", left_on=["productid", "imageid"], right_on=["productid", "imageid"])
    train["duplicated"] = train.duplicated(subset="hash") # ajout d'une colonne pour identifier les doublons

    plt.figure(figsize=(12,10))

    # Visualisation de la distribution des images en fonction de leur niveau de luminosité et de contraste
    plt.subplot(221)
    sns.boxplot(train["mean_luminosity"])
    plt.ylabel("Niveau de luminosité")
    plt.title("Distibution niveau de luminosité des images")

    plt.subplot(222)
    sns.boxplot(train["mean_stddev_luminosity"])
    plt.ylabel("Niveau de contraste")
    plt.title("Distibution du niveau de contraste des images")

    plt.savefig(os.path.join(save_dir_path, "boxplots_luminosity_contrast.png"))

    plt.show()

    # Visualisation de la distriution des doublons d'images par catégorie cible
    duplicated = train.groupby("prdtypecode")["duplicated"].sum()
    duplicated_normalized = train.groupby("prdtypecode")["duplicated"].mean()

    plt.figure(figsize=(12,10))
    plt.subplot(211)

    # Nombre de doublon d'image par catégorie cible
    sns.barplot(y=duplicated, x=duplicated.index)
    plt.title("Nb de doublons par catégorie cible")
    plt.xlabel("Catégories")
    plt.ylabel("Nb doublons")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.subplot(212)

    # Nombre de doublon d'image par catégorie cible
    sns.barplot(y=duplicated_normalized, x=duplicated_normalized.index)
    plt.title("% de doublons par catégorie cible")
    plt.xlabel("Catégories")
    plt.ylabel("% doublons")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir_path, "barplot_repartition_doublon.png"))

    plt.show()