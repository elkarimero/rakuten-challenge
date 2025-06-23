import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import cv2

from tensorflow.keras.applications.efficientnet import preprocess_input
from preprocessing.image_preprocessing import zoom_picture

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

categories = {
            10: "Livres occasion",
            2280: "Journaux et revues occasions",
            2403: "Livres, BD et magazines",
            2522: "Fournitures papeterie et accessoires bureau",
            2705: "Livres neufs",
            40: "Jeux videos, CDs, équipements, câbles, neufs",
            50: "Accessoires gaming",
            60: "Consoles de jeux",
            2462: "Jeux vidéos occasion",
            2905: "Jeux vidéos pour PC",
            1140: "Figurines, objets pop culture",
            1160: "Cartes de jeux",
            1180: "Figurines et jeux de rôles",
            1280: "Jouets enfant",
            1281: "Jeux société enfants",
            1300: "Modélisme",
            1302: "Jeux de pleins air, Habits",
            1560: "Mobilier général",
            2582: "Mobilier de jardin",
            1320: "Puériculture, accessoire bébé",
            2220: "Animalerie",
            2583: "Piscine et accessoires",
            2585: "Outillages de jardin, équipements extérieur et piscine",
            1920: "Linge de maison",
            2060: "Décoration",
            1301: "Chaussettes bébés, petites photos",
            1940: "Confiserie"
        }

# Fonction de prétraitement d'image
def preprocess_image(image: Image.Image):
    #image = tf.io.read_file(filepath)
    #image = tf.image.decode_image(image, channels=3)
    #image.set_shape([None, None, 3])

    # Convertir PIL en array NumPy
    image_np = np.array(image)
    # Convertir RGB (PIL) en BGR (OpenCV)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    _ , _ , _ , _ , _ , zoomed = zoom_picture(img_src=image_cv2)  # Nettoyage de l'image
    resized = tf.image.resize(zoomed, (224, 224))
    
    # Normalisation pour EfficientNetB0 ([-1, 1] si preprocess_input est utilisé)
    image = preprocess_input(resized)
    
    return tf.expand_dims(image, axis=0)

# Interface Streamlit
st.title("🖼️ Démonstration - EfficientNetB0 Fine-tuné")

uploaded_file = st.file_uploader("📤 Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="🖼️ Image chargée", width=200)

    if st.button("🔍 Prédire"):
        with st.spinner("Prédiction en cours..."):
            input_tensor = preprocess_image(image)
            predictions = model.predict(input_tensor)[0]
            top_class = np.argmax(predictions)
            confidence = predictions[top_class] * 100
            # Convertir la classe prédite en étiquette
            predictions = {categories.get(int(label_encoder.inverse_transform([i])[0]), "Inconnu"): float(pred) for i, pred in enumerate(predictions)}
            top_class = label_encoder.inverse_transform([top_class])[0]  # Décoder la classe prédite
            top_cat = categories.get(int(top_class), "Inconnu")  # Obtenir le nom de la catégorie

            st.success(f"✅ Classe prédite : {top_cat} - {top_class} avec une confiance de {confidence:.2f}%")
            st.bar_chart(predictions)


