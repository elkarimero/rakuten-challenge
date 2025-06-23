import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.title("Présentation et exploration des datasets")
st.write("""
lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
""")

# Chargement des données
X_train = pd.read_csv("./data/raw/X_train.csv", sep=",",index_col=0)
y_train = pd.read_csv("./data/raw/Y_train.csv", sep=",",index_col=0)
image_train = pd.read_csv("./data/raw/image_train.csv", sep=",",index_col=0)
# merge des dataframe pour faciliter l'exploration
train = pd.concat([X_train, y_train], axis=1)
train = pd.merge(train, image_train, how="inner", left_on=["productid", "imageid"], right_on=["productid", "imageid"])
train["duplicated"] = train.duplicated(subset="hash") # ajout d'une colonne pour identifier les doublons

explo_text_tab, explo_image_tab = st.tabs(["Données textuelles", "Images"])

with explo_text_tab:
    st.header("Exploration des données textuelles")
    
with explo_image_tab:

    with st.expander("Analyse exploratoire des images", expanded=True):
        col1, col2 = st.columns([2, 1]) 
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

        col2.subheader("Luminosité et du contraste")

        ####
        # Analyse de la répartition des doublons
        ####

        col1, col2 = st.columns([2, 1]) 
 
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

        col2.subheader("Répartition des doublons")
    
    with st.expander("Exemples d'images problématiques"):
        st.markdown('''
            - Des **doublons :** *des images très similaires, voire identiques (catégorie 2583 - 'Piscine et accessoires')*
            - Des **placeholders :** *images remplaçant des images produit manquantes (pratique courante)*
            - Des **objets trop petits :** *objet d’intérêt occupe très peu de pixels par rapport à l’ensemble de l’image*
        ''')

        col1_image, col2_image, col3_image, col4_image = st.columns(4)

        # première ligne = vrais doublons
        col1_image.image("./images/doublon_a1.jpg", width=200)
        col2_image.image("./images/doublon_a2.jpg", width=200)
        col3_image.image("./images/doublon_b1.jpg", width=200)
        col4_image.image("./images/doublon_b2.jpg", width=200)
        # deuxième ligne = doublons avec placeholder
        col1_image.image("./images/placeholder1.jpg", width=200)
        col2_image.image("./images/placeholder2.jpg", width=200)
        col3_image.image("./images/placeholder3.jpg", width=200)
        col4_image.image("./images/placeholder4.jpg", width=200)
        # troisième ligne = objets problématiques
        col1_image.image("./images/small1.jpg", width=200)
        col2_image.image("./images/small2.jpg", width=200)
        col3_image.image("./images/small3.jpg", width=200)
        col4_image.image("./images/small4.jpg", width=200)
    
    