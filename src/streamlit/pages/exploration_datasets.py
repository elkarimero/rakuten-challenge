import streamlit as st

st.title("Présentation et exploration des datasets")
st.write("""
lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
""")


explo_text_tab, explo_image_tab = st.tabs(["Données textuelles", "Images"])

with explo_text_tab:
    st.header("Exploration des données textuelles")
    
with explo_image_tab:

    with st.expander("Analyse exploratoire des images", expanded=True):
        col1, col2 = st.columns(2)

        col1.subheader("Analyse de la luminosité et du contraste")
        col1.image("./images/boxplots_luminosity_contrast.png")


        col2.subheader("Analyse de la répartition des doublons")
        col2.image("./images/barplot_repartition_doublon.png")

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
