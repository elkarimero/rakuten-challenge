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

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Chargement et prétraitement d'une image depuis son chemin
def load_and_preprocess_image(filepath):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (224, 224))
    
    # Normalisation pour EfficientNetB0 ([-1, 1] si preprocess_input est utilisé)
    from tensorflow.keras.applications.efficientnet import preprocess_input
    image = preprocess_input(image)
    
    return image

# Fusion des prédictions image + texte
def late_fusion_predict(text_model, image_model, text_input, image_path, alpha=0.5):
    # Texte → SVM
    prob_text = text_model.predict([text_input])[0]

    # Image → EfficientNet
    image_tensor = load_and_preprocess_image(image_path)
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # Add batch dimension
    prob_image = image_model.predict(image_tensor, verbose=0)[0]

    # Fusion
    prob_combined = alpha * prob_image + (1 - alpha) * prob_text
    predicted_class = np.argmax(prob_combined)

    return predicted_class, prob_combined

# Évaluation globale
def evaluate_fusion(text_model, image_model, X_text, image_paths, y_true, alpha=0.5):
    y_pred = []
    for text_input, image_path in zip(X_text, image_paths):
        pred_class, _ = late_fusion_predict(text_model, image_model, text_input, image_path, alpha)
        y_pred.append(pred_class)

    acc = accuracy_score(y_true, y_pred)
    return acc, y_pred


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
#text_model = joblib.load('./models/inference_pipeline_svc_model.joblib')
text_model = joblib.load('./models/svm2.pkl')
vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")

# Interface Streamlit
#st.title("🖼️ Démonstration - EfficientNetB0 Fine-tuné")

form_cols = st.columns(2)
uploaded_file   = form_cols[0].file_uploader("📤 Choisissez une image", type=["jpg", "jpeg", "png"])
text_input      = form_cols[1].text_area("📝 Entrez une description du produit")


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    #form_cols[0].image(image, caption="Image chargée", width=100)

if st.button("🔍 Prédire"):
    #st.subheader("Performances du modèle avec et sans fine-tuning")
    fusion = st.columns(1)[0]

    col1, spacer, col2 = st.columns([1, 0.1, 1])

    with st.spinner("Prédiction en cours..."):

        if uploaded_file:
            # Prédiction de l'image
            input_tensor, img_preprocessed = preprocess_image(image)
            predictions = model.predict(input_tensor)[0]
            top_class = np.argmax(predictions)
            confidence = predictions[top_class] * 100
            # Convertir la classe prédite en étiquette
            img_proba = {categories.get(int(label_encoder.inverse_transform([i])[0]), "Inconnu"): float(pred) for i, pred in enumerate(predictions)}
            top_class = label_encoder.inverse_transform([top_class])[0]  # Décoder la classe prédite
            top_cat = categories.get(int(top_class), "Inconnu")  # Obtenir le nom de la catégorie

            # Afficher les résultats
            
            #col1.success(f"✅ Classe prédite : {top_cat} - {top_class} avec une confiance de {confidence:.2f}%")
            col1.success(f"📸 Classe prédite : {top_cat} - {top_class}  - confiance : {confidence:.2f}%")
            col1.bar_chart(img_proba, horizontal=True)

            # Afficher le grad-CAM
            cols = col1.columns(3)
            grad_cam_image, predicted_class, predicted_score = grad_cam(input_tensor, model, model.get_layer("top_conv"))
            # Afficher l'image originale
            cols[0].image(image, caption="Image chargée", width=224)
            # Afficher l'image prétraitée
            cols[1].image(cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB), caption="Image traitée", width=224)
            # Afficher l'image avec Grad-CAM
            cols[2].image(grad_cam_image, caption="Image avec Grad-CAM", width=224)

        # Prédiction du texte
        if text_input:
            text_input = [text_input]
            text_input_vectorized = vectorizer.transform(text_input).toarray() 
            text_predictions = text_model.predict_proba(text_input_vectorized)[0]
            text_top_class = np.argmax(text_predictions)
            text_confidence = text_predictions[text_top_class] * 100
            text_proba = {categories.get(int(label_encoder.inverse_transform([i])[0]), "Inconnu"): float(pred) for i, pred in enumerate(text_predictions)}
            text_top_class = label_encoder.inverse_transform([text_top_class])[0]  # Décoder la classe prédite
            text_top_cat = categories.get(int(text_top_class), "Inconnu")
            #col2.success(f"✅ Classe prédite pour le texte : {text_top_cat} - {text_top_class} avec une confiance de {text_confidence:.2f}%")
            col2.success(f"📝 Classe prédite : {text_top_cat} - {text_top_class} - confiance : {text_confidence:.2f}%")
            col2.bar_chart(text_proba, horizontal=True)

        
        if text_input and uploaded_file:    
            # Fusion des prédictions
            alpha = 0.5
            prob_combined = alpha * predictions + (1 - alpha) * text_predictions
            predicted_class = np.argmax(prob_combined)
            combined_confidence = np.max(prob_combined) * 100
            predicted_class_label = label_encoder.inverse_transform([predicted_class])[0]

            predicted_category = categories.get(int(predicted_class_label), "Inconnu")
            fusion.success(f"✅ Classe prédite par fusion : {predicted_category} - {predicted_class_label} avec une confiance de {combined_confidence:.2f}%")

