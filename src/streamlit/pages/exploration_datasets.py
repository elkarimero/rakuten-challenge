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
    
    with st.expander("Traitements des images problématiques"):
        st.markdown('''
            - **Nuances de gris :** *On convertit les images en nuances de gris pour réduire la complexité des données et se concentrer sur les contours des objets*
            - **Binarisation :** *On applique un seuillage pour convertir l'image en noir et blanc, ce qui permet de mieux détecter les contours des objets et de réduire le bruit*     
            - **Contours :** *On utilise la détection de contours pour identifier les objets dans l'image, ce qui permet de mieux les isoler et de réduire le bruit*
            - **Bounding box :** *On dessine une boîte englobante autour de l'objet détecté pour mieux visualiser l'objet d'intérêt*
            - **Zoom :** *On redimensionne l'image pour se concentrer sur l'objet d'intérêt, ce qui permet de mieux le visualiser et de réduire la taille de l'image*
        ''')

        col1_image, col2_image, col3_image, col4_image, col5_image, col6_image = st.columns(6)
        col1_image.subheader("Originale")
        col2_image.subheader("Nuances de gris")
        col3_image.subheader("Binarisée")
        col4_image.subheader("Contours")    
        col5_image.subheader("Bounding box") 
        col6_image.subheader("Zoom")

        col1_image.image("./images/img0_orig.jpg", width=200)
        col2_image.image("./images/img0_gray.jpg", width=200)
        col3_image.image("./images/img0_binary.jpg", width=200)
        col4_image.image("./images/img0_contoured.jpg", width=200)
        col5_image.image("./images/img0_rectangle.jpg", width=200)
        col6_image.image("./images/img0_resized.jpg", width=200)

        col1_image.image("./images/img1_orig.jpg", width=200)
        col2_image.image("./images/img1_gray.jpg", width=200)
        col3_image.image("./images/img1_binary.jpg", width=200)
        col4_image.image("./images/img1_contoured.jpg", width=200)
        col5_image.image("./images/img1_rectangle.jpg", width=200)
        col6_image.image("./images/img1_resized.jpg", width=200)
       
        col1_image.image("./images/img2_orig.jpg", width=200)
        col2_image.image("./images/img2_gray.jpg", width=200)
        col3_image.image("./images/img2_binary.jpg", width=200)
        col4_image.image("./images/img2_contoured.jpg", width=200)
        col5_image.image("./images/img2_rectangle.jpg", width=200)
        col6_image.image("./images/img2_resized.jpg", width=200)

        col1_image.image("./images/img3_orig.jpg", width=200)
        col2_image.image("./images/img3_gray.jpg", width=200)
        col3_image.image("./images/img3_binary.jpg", width=200)
        col4_image.image("./images/img3_contoured.jpg", width=200)
        col5_image.image("./images/img3_rectangle.jpg", width=200)
        col6_image.image("./images/img3_resized.jpg", width=200)

        #col1_image.image("./images/img4_orig.jpg", width=200)
        #col2_image.image("./images/img4_gray.jpg", width=200)
        #col3_image.image("./images/img4_binary.jpg", width=200)
        #col4_image.image("./images/img4_contoured.jpg", width=200)
        #col5_image.image("./images/img4_rectangle.jpg", width=200)
        #col6_image.image("./images/img4_resized.jpg", width=200)

       

