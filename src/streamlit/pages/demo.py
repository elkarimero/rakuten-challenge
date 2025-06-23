import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import cv2
import matplotlib.pyplot as plt
from preprocessing.image_preprocessing import preprocess_image
from visualization.grad_cam import grad_cam
from data.constants import categories

st.title("Démonstration interactive")
st.write("""
Bienvenue dans notre projet de data science.
Ce tableau de bord interactif vous guide à travers les étapes du projet.
""")

# Charger le modèle
@st.cache_resource
def load_model(nb_class = 27):
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    # Geler d'abord le modèle de base
    base_model.trainable = False 

    # Couches de classification 
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(nb_class, activation='softmax')(x)

    # Construction finale 
    model = tf.keras.Model(base_model.input, outputs)

    model.load_weights("./models/EfficientNetB0_model_finetuned_best.weights.h5")

    return model

model = load_model()

# Chargement de l'encoder
label_encoder = joblib.load('./models/label_encoder.joblib')

# Interface Streamlit
st.title("🖼️ Démonstration - EfficientNetB0 Fine-tuné")

uploaded_file = st.file_uploader("📤 Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Image chargée", width=100)

    if st.button("🔍 Prédire"):
        with st.spinner("Prédiction en cours..."):
            input_tensor, img_preprocessed = preprocess_image(image)
            predictions = model.predict(input_tensor)[0]
            top_class = np.argmax(predictions)
            confidence = predictions[top_class] * 100
            # Convertir la classe prédite en étiquette
            predictions = {categories.get(int(label_encoder.inverse_transform([i])[0]), "Inconnu"): float(pred) for i, pred in enumerate(predictions)}
            top_class = label_encoder.inverse_transform([top_class])[0]  # Décoder la classe prédite
            top_cat = categories.get(int(top_class), "Inconnu")  # Obtenir le nom de la catégorie

            # Afficher les résultats
            cols = st.columns(6)
            grad_cam_image, predicted_class, predicted_score = grad_cam(input_tensor, model, model.get_layer("top_conv"))
            # Afficher l'image originale
            cols[0].image(image, caption="Image chargée", width=224)
            # Afficher l'image prétraitée
            cols[1].image(cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB), caption="Image traitée", width=224)
            # Afficher l'image avec Grad-CAM
            cols[2].image(grad_cam_image, caption="Image avec Grad-CAM", width=224)
            st.success(f"✅ Classe prédite : {top_cat} - {top_class} avec une confiance de {confidence:.2f}%")
            st.bar_chart(predictions)


