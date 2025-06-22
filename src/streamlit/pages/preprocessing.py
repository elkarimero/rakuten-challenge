import streamlit as st

st.title("Méthodologie et préparation des données")
st.write("""
lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
""")


explo_text_tab, explo_image_tab = st.tabs(["Données textuelles", "Images"])

with explo_text_tab:
    st.header("Exploration des données textuelles")
    
with explo_image_tab:
    with st.expander("Traitements des images problématiques"):

        st.subheader("Centrage et standardisation des images")
        st.markdown('''
            - **Nuances de gris :** *On convertit les images en nuances de gris pour réduire la complexité des données et se concentrer sur les contours des objets*
            - **Binarisation :** *On applique un seuillage pour convertir l'image en noir et blanc, ce qui permet de mieux détecter les contours des objets et de réduire le bruit*     
            - **Contours :** *On utilise la détection de contours pour identifier les objets dans l'image, ce qui permet de mieux les isoler et de réduire le bruit*
            - **Bounding box :** *On dessine une boîte englobante autour de l'objet détecté pour mieux visualiser l'objet d'intérêt*
            - **Zoom :** *On redimensionne l'image pour se concentrer sur l'objet d'intérêt, ce qui permet de mieux le visualiser et de réduire la taille de l'image*
        ''')
        import cv2
        from preprocessing.image_preprocessing import zoom_picture

        col1_image, col2_image, col3_image, col4_image, col5_image, col6_image = st.columns(6)

        col1_image.subheader("Originale")
        col2_image.subheader("Nuances de gris")
        col3_image.subheader("Binarisée")
        col4_image.subheader("Contours")    
        col5_image.subheader("Bounding box") 
        col6_image.subheader("Zoom")

        img_files = []
        img_files.append("./images/img0_orig.jpg")
        img_files.append("./images/img1_orig.jpg")
        img_files.append("./images/img2_orig.jpg")
        img_files.append("./images/img3_orig.jpg")
        #img_files.append("./images/img4_orig.jpg")

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
        
    with st.expander("Rééquilibrage et augmentation des données"):

        st.markdown('''
            - **Rééquilibrage :** *Utilisation de **la médiane** comme seuil d’équilibrage*
            - **Augmentation :** *Utiliisation de techniques d'augmentation des données pour générer de nouvelles images à partir des images existantes, en appliquant des transformations telles que la rotation, le zoom, le retournement, etc.*
        ''')

        import streamlit as st
        import tensorflow as tf
        import numpy as np
        from PIL import Image

        flip = st.checkbox("Flip horizontal", value=True)
        rotation = st.slider("Rotation (± fraction)", 0.0, 0.5, 0.2, step=0.01)
        zoom = st.slider("Zoom (± fraction)", 0.0, 0.5, 0.2, step=0.01)
        contrast = st.slider("Contraste (± %)", 0.0, 1.0, 0.2, step=0.01)
        
        for _, image in enumerate(final_images):
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
