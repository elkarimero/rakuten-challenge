import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from path_config import get_data_path, get_image_path

st.title("Présentation et exploration des datasets")
st.write("Dans cette partie, nous présentons les différentes difficultés présentées par les datasets, images et texte.")

# Chargement des données
X_train = pd.read_csv(get_data_path("raw/X_train.csv"), sep=",", index_col=0)
y_train = pd.read_csv(get_data_path("raw/Y_train.csv"), sep=",", index_col=0)
image_train = pd.read_csv(get_data_path("raw/image_train.csv"), sep=",", index_col=0)

# Merge des dataframes pour faciliter l'exploration
train = pd.concat([X_train, y_train], axis=1)
train = pd.merge(train, image_train, how="inner", left_on=["productid", "imageid"], right_on=["productid", "imageid"])
train["duplicated"] = train.duplicated(subset="hash")  # Ajout d'une colonne pour identifier les doublons

explo_text_tab, explo_image_tab = st.tabs(["Données textuelles", "Exploration du dataset d'images"])

with explo_text_tab:
    st.header("Analyse exploratoire textuelle")

    # Ajout d'une section pour décrire les particularités du jeu de données
    st.subheader("Particularités du jeu de données textuelles")
    st.write("""
    Le jeu de données présente plusieurs particularités qui ont nécessité une attention particulière en phase de préparation :
    - **Redondance des colonnes** : Les colonnes "dénomination" et "description" étaient souvent redondantes, avec parfois un simple copier-coller entre les deux.
    - **Langues multiples** : Certaines descriptions étaient rédigées en anglais ou dans d'autres langues, ce qui complexifie le traitement sémantique.
    - **Artefacts textuels** : Les textes contenaient de nombreux artefacts tels que des balises HTML, des caractères spéciaux, des accents mal encodés ou d'autres bruits textuels nuisant à l’analyse.
    """)

    # Ajout d'une section pour l'analyse textuelle
    st.subheader("Analyse des données textuelles")

    # Exemple de texte
    st.write("Exemple de texte brut :")
    st.write(train['designation'].iloc[0])

    # Ajout d'une analyse de texte
    st.subheader("Analyse de la distribution des longueurs de texte")

    # Calcul de la longueur des textes
    train['text_length'] = train['designation'].apply(len)

    # Création d'un histogramme des longueurs de texte avec une taille réduite
    fig, ax = plt.subplots(figsize=(6, 3))  # Taille réduite
    sns.histplot(train['text_length'], bins=50, ax=ax)
    ax.set_title("Distribution des longueurs de texte")
    ax.set_xlabel("Longueur du texte")
    ax.set_ylabel("Fréquence")

    st.pyplot(fig)

    # Ajout d'une analyse des mots fréquents
    st.subheader("Analyse des mots fréquents")

    from collections import Counter
    from wordcloud import WordCloud

    # Concaténation de tous les textes
    all_text = " ".join(designation for designation in train['designation'])

    # Création d'un nuage de mots avec une taille réduite
    wordcloud = WordCloud(width=600, height=300, background_color='white').generate(all_text)

    # Affichage du nuage de mots
    fig, ax = plt.subplots(figsize=(8, 4))  # Taille réduite
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")

    st.pyplot(fig)


with explo_image_tab:

    with st.expander("Analyse exploratoire des images", expanded=True):
        col1, spacer, col2 = st.columns([1, 0.1, 1])
        col1.subheader(" ")

        ####
        # Analyse de la luminosité et du contraste des images
        ####
        # Création de la figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Premier boxplot : luminosité
        sns.boxplot(train["mean_luminosity"], ax=axs[0])
        axs[0].set_ylabel("Niveau de luminosité")
        axs[0].set_title("Distribution du niveau de luminosité des images")

        # Deuxième boxplot : contraste
        sns.boxplot(train["mean_stddev_luminosity"], ax=axs[1])
        axs[1].set_ylabel("Niveau de contraste")
        axs[1].set_title("Distribution du niveau de contraste des images")

        # Affichage avec Streamlit
        col1.pyplot(fig)

        ####
        # Analyse de la répartition des doublons
        ####
 
        # Agrégations
        duplicated = train.groupby("prdtypecode")["duplicated"].sum()
        duplicated_normalized = train.groupby("prdtypecode")["duplicated"].mean()

        # Création de la figure et des sous-graphiques
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Barplot : nombre de doublons par catégorie
        sns.barplot(y=duplicated.values, x=duplicated.index, ax=axs[0])
        axs[0].set_title("Nb de doublons par catégorie cible")
        axs[0].set_xlabel("Catégories")
        axs[0].set_ylabel("Nb doublons")
        axs[0].tick_params(axis='x', rotation=45)

        # Barplot : % de doublons par catégorie
        sns.barplot(y=duplicated_normalized.values, x=duplicated_normalized.index, ax=axs[1])
        axs[1].set_title("% de doublons par catégorie cible")
        axs[1].set_xlabel("Catégories")
        axs[1].set_ylabel("% doublons")
        axs[1].tick_params(axis='x', rotation=45)

        # Mise en page et affichage dans Streamlit
        fig.tight_layout()
        col1.pyplot(fig)

        ####
        col2.subheader(" ")
        col2.subheader("Analyse des caractéristiques des images")
        col2.markdown('''
            - **Dimensions** des images (hauteur / largeur / ratio)
            - Niveaux moyen de **Luminosité et de contraste**
            - **Taux de doublons par catégorie** en utilisant des Hash perceptuels
        ''')
        col2.subheader("Conclusion sur la qualité des images")
        col2.markdown('''
            - Une majorité des images avec **luminosité moyenne très élevée** (médiane > 200)  
            - Quelques cas abérants de **luminosité très faible** (inférieur à 50)
            - **Contraste correct** dans l'ensemble mais **quelques cas abérants** de contraste très faible (inférieur à 5)
            - Plusieurs catégories avec un fort taux de doublons (15 à 30% de doublons)
        ''')


    col_img_pb = st.columns(2)

    col_img_pb[0].subheader("Analyse des images problématiques")

    col_img_pb[0].markdown('''
        - Des **objets trop petits ->** *objet d’intérêt occupe très peu de pixels par rapport à l’ensemble de l’image*
        - Des **placeholders ->** *images remplaçant des images produit manquantes (pratique courante)*
        - Des **doublons ->** *des images très similaires, voire identiques (catégorie 2583 - 'Piscine et accessoires')*
        - Des **images non exploitables ->** *(ex: images avec contraste trop faible, quasi monochromes ...)*
    ''')

    col_img_pb[0].subheader("Traitements envisagés")
    col_img_pb[0].markdown('''
        - Création d'un pipeline de preprocessing pour zoomer les images avec des objets trop petit
        - Suppression des images pouvant impacter négativement les performances
            - Placeholders 
            - Images de non exploitables
            - Doublon
    ''')

    
    with col_img_pb[1].expander("Exemples d'images problématiques"):
        col_images = st.columns(3)
        
        # Images trop petites
        col_images[0].image(get_image_path("small1.jpg"), width=200, caption="Objet trop petit")

        # placeholder
        col_images[1].image(get_image_path("placeholder1.jpg"), width=200, caption="Placeholder")
        col_images[2].image(get_image_path("placeholder2.jpg"), width=200, caption="Placeholder")

        col_images[0].image(get_image_path("monochrome.jpg"), width=200,  caption="Quasi monochrome")
        # vrais doublons
        col_images[1].image(get_image_path("doublon_a1.jpg"), width=200, caption="Doublon")
        col_images[2].image(get_image_path("doublon_a2.jpg"), width=200, caption="Doublon")
    

        
    
    