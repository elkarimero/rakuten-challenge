import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

from tensorflow.keras.applications.efficientnet import preprocess_input

st.title("Démonstration interactive")
st.write("""
Bienvenue dans notre projet de data science.
Ce tableau de bord interactif vous guide à travers les étapes du projet.
""")

# Nombre de classes dans votre modèle
nb_class = 27  # Changez ce nombre selon votre cas

# Charger le modèle
#@st.cache_resource
def load_model():
    image_model = tf.keras.models.load_model('./models/EfficientNetB0_model_finetuned_best.keras')

    return image_model

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

class_orig = ['10',
 '1140',
 '1160',
 '1180',
 '1280',
 '1281',
 '1300',
 '1301',
 '1302',
 '1320',
 '1560',
 '1920',
 '1940',
 '2060',
 '2220',
 '2280',
 '2403',
 '2462',
 '2522',
 '2582',
 '2583',
 '2585',
 '2705',
 '2905',
 '40',
 '50',
 '60']


# Fonction de prétraitement d'image
def preprocess_image(image: Image.Image):
    #image = tf.io.read_file(filepath)
    #image = tf.image.decode_image(image, channels=3)
    #image.set_shape([None, None, 3])
    image = tf.image.resize(image, (224, 224))
    
    # Normalisation pour EfficientNetB0 ([-1, 1] si preprocess_input est utilisé)
    image = preprocess_input(image)
    
    return tf.expand_dims(image, axis=0)

# Interface Streamlit
st.title("🧠 Démonstration - EfficientNetB0 Fine-tuné")

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
            predictions = {categories.get(int(class_orig[i]), "Inconnu"): float(pred) for i, pred in enumerate(predictions)}
            top_class = int(class_orig[top_class]) # Décoder la classe prédite
            top_cat = categories.get(top_class, "Inconnu")  # Obtenir le nom de la catégorie

            st.success(f"✅ Classe prédite : {top_cat} {top_class} avec une confiance de {confidence:.2f}%")
            st.bar_chart(predictions)


