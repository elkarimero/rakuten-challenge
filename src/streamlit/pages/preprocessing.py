import streamlit as st

st.title("Méthodologie et préparation des données")

explo_text_tab, preprocessing_image_tab, augmentation_image_tag = st.tabs(["Données textuelles", "Preprocessing des images", "Augmentation des images"])

with explo_text_tab:
    with st.expander("**Pipeline du traitement de texte**", expanded=True):
        st.markdown('''
                    - **Suppression des doublons** : *sur les colonnes (dénomination et description) et sur les lignes (réitérée après chaque étape)*
                    - **Fusion des informations** : *fusion des colonnes dénomination et description* 
                    - **Suppression des éléments web** : *suppression des URLs ainsi que des balises HTML*
                    - **Normalisation Unicode** : *normalisation en minuscules, sans accent, sans caractères spéciaux et sans espace inutile*
                    - **Traduction vers le français** : *traduction à l'aide de l'API gratuite DeepL*
                    - **Traitement lexical** : *Tokenisation, suppression des Stopwords et Lemmatisation*
                    - **Equilibrage des classes**
            ''')

    with st.expander("**Quelques graphiques**", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("Plots/pourcentage_langues.png")
        with col2:
            st.image("Plots/Equilibrage_avant.png")
        with col3:
            st.image("Plots/Equilibrage_apres.png")    
    
        
    
with preprocessing_image_tab:
    st.subheader("Pipeline de prétraitement des images")
    st.markdown("**Objectif :** *Centrer l'image sur l'objet d'intérêt et la redimensionner pour optimiser la performance du modèle de classification*")
    st.markdown('''
            1. **Nuances de gris** -> *On convertit les images en nuances de gris pour réduire la complexité des données et se concentrer sur les contours des objets*
            2. **Binarisation** -> *On applique un seuillage pour convertir l'image en noir et blanc, ce qui permet de mieux détecter les contours des objets et de réduire le bruit*     
            3. **Contours** -> *On utilise la détection de contours pour identifier les objets dans l'image, ce qui permet de mieux les isoler*
            4. **Bounding box** -> *On dessine une boîte englobante autour de l'objet détecté pour mieux visualiser l'objet d'intérêt*
            5. **Zoom** -> *On redimensionne l'image en zoomant sur l'objet d'intérêt, ce qui permet de mieux le visualiser et d'optimiser la taille de l'image pour l'entrainement (224x224)*
        ''')
    
    with st.expander("Illustration du pipeline"):
        import cv2
        from preprocessing.image_preprocessing import zoom_picture

        col1_image, col2_image, col3_image, col4_image, col5_image, col6_image = st.columns(6)

        col1_image.text("Originale")
        col2_image.text("Nuances de gris")
        col3_image.text("Binarisée")
        col4_image.text("Contours")    
        col5_image.text("Bounding box") 
        col6_image.text("Zoom")

        img_files = []
        #img_files.append("./images/img0_orig.jpg")
        img_files.append("./images/img1_orig.jpg")
        img_files.append("./images/img2_orig.jpg")
        #img_files.append("./images/img3_orig.jpg")
        #img_files.append("./images/small5.jpg")

        final_images = []
        for i, filepath in enumerate(img_files):
            img_orig, gray, binary, contoured, rectangle, resized = zoom_picture(filepath=filepath)
            col1_image.image(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB), width=200)
            col2_image.image(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), width=200)
            col3_image.image(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB), width=200)
            col4_image.image(cv2.cvtColor(contoured, cv2.COLOR_BGR2RGB), width=200)
            col5_image.image(cv2.cvtColor(rectangle, cv2.COLOR_BGR2RGB), width=200)
            col6_image.image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), width=200)
            final_images.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

with augmentation_image_tag:
    st.subheader("**Rééquilibrage et augmentation des données**")
    st.markdown("**Objectif :** *Augmenter la diversité des données d'entraînement pour améliorer la robustesse du modèle de classification*")
    st.markdown('''
            - **Rééquilibrage** *des classes autour de la* **médiane** 
                - *suppression des doublons très fréquents dans les classes majoritaires*
                - *duplication des images de classes minoritaires*
            - *Génération de nouvelles images à partir des images existantes, en appliquant des transformations telles
            que* **le retournement horizontal, la rotation, le zoom, le contraste.**
        ''')
    with st.expander("Exemple d'augmentation des données"):

        import streamlit as st
        import tensorflow as tf
        import numpy as np
        from PIL import Image

        col_form1, col_form2, col_form3, col_form4  = st.columns([0.7,1.1,1.1,1.1])
        flip = col_form1.checkbox("Flip horizontal", value=True)
        rotation = col_form2.slider("Rotation (± fraction)", 0.0, 1.0, 0.2, step=0.1)
        zoom = col_form3.slider("Zoom (± fraction)", 0.0, 1.0, 0.4, step=0.1)
        contrast = col_form4.slider("Contraste (± %)", 0.0, 1.0, 0.2, step=0.1)
        
        for _, image in enumerate(final_images[0:1]):
            # Prétraitement
            img = cv2.resize(image, (224, 224))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            img_array = tf.expand_dims(img_array, 0)

            # Pipeline d'augmentation dynamique
            augmentation_layers = []

            if flip:
                augmentation_layers.append(tf.keras.layers.RandomFlip("horizontal"))
            if rotation > 0:
                augmentation_layers.append(tf.keras.layers.RandomRotation(factor=rotation))
            if zoom > 0:
                augmentation_layers.append(tf.keras.layers.RandomZoom(height_factor=zoom, width_factor=zoom))
            if contrast > 0:
                augmentation_layers.append(tf.keras.layers.RandomContrast(factor=contrast))

            data_augmentation = tf.keras.Sequential(augmentation_layers)

            # Génération et affichage des images augmentées
            cols = st.columns(6)
            cols[0].image(img, width=200, caption="Image originale")
            for i in range(1,6):
                aug_img = data_augmentation(img_array, training=True)[0].numpy()
                aug_img = np.clip(aug_img * 255, 0, 255).astype(np.uint8)
                with cols[i]:
                    st.image(aug_img, width=200, caption=f"Augmentation {i+1}")
