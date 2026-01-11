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
from path_config import get_model_path, get_data_path
import pandas as pd


st.title("Démonstrateur interactive")


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

    model.load_weights(get_model_path("EfficientNetB0_model_finetuned_best.weights.h5"))

    return model

model = load_model()

# Chargement de l'encoder
label_encoder = joblib.load(get_model_path('label_encoder.joblib'))
#text_model = joblib.load(get_model_path('inference_pipeline_svc_model.joblib'))
text_model = joblib.load(get_model_path('svm2.pkl'))
vectorizer = joblib.load(get_model_path("tfidf_vectorizer.pkl"))

# Interface Streamlit

# Initialisation dans session_state (si pas déjà fait)
if "description" not in st.session_state:
    st.session_state["description"] = ""

# Initialisation dans session_state (si pas déjà fait)
if "default_image" not in st.session_state:
    st.session_state["default_image"] = None

def random_product():
    df = pd.read_csv(get_data_path("test/train_clean.csv"))
    df_sample = df.sample(n=1)
    sample = df_sample.iloc[0]
    return sample["merged"], get_data_path(f"test/images_train/{sample['filename']}")

menu_cols = st.columns(8)
# Bouton pour initialiser avec des valeurs dynamiques
if menu_cols[0].button("⚡ Charger"):
    st.session_state["description"], st.session_state["default_image"] = random_product()

# Bouton pour réinitialiser les champs
if menu_cols[1].button("🔄 Réinitialiser"):
    st.session_state["description"] = ""
    st.session_state["default_image"] = None

predict = menu_cols[2].button("🔍 Prédire")

# Formulaire de saisie
form_cols = st.columns([1.5,0.2,0.8])
text_input      = form_cols[0].text_area("📝 Entrez une description du produit", value=st.session_state["description"], height = 300)
file_uploader   = form_cols[0].file_uploader("📸 Choisissez une image", type=["jpg", "jpeg", "png"])

# Mise à jour du contenu de la session state quand le textarea change
st.session_state.textarea_content = text_input

# data par défaut
#image_defaut = Image.open("./images/img4_orig.jpg").convert("RGB")

if file_uploader:
    image = Image.open(file_uploader).convert("RGB")
    uploaded_file = True
elif st.session_state.default_image is not None:
    image = Image.open(st.session_state.default_image).convert("RGB")
    uploaded_file = True
else:
    image = None
    uploaded_file = False

if predict:
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

        if text_input and uploaded_file:    
            # Fusion des prédictions
            alpha = 0.5
            prob_combined = alpha * predictions + (1 - alpha) * text_predictions
            predicted_class = np.argmax(prob_combined)
            combined_confidence = np.max(prob_combined) * 100
            predicted_class_label = label_encoder.inverse_transform([predicted_class])[0]

            predicted_category = categories.get(int(predicted_class_label), "Inconnu")

            # Afficher les résultats de la fusion
            form_cols[2].metric(f"📸 + 📝 Classe prédite par fusion :", f"{predicted_category} - {predicted_class_label}")
            form_cols[2].metric("💪 Confiance", f"{combined_confidence:.2f}%")
        
        if uploaded_file:
            form_cols[2].image(image, caption="Image chargée", width=224)

        with st.expander("**Interprétabilité**"):

            if text_input and not uploaded_file:
                form_cols[2].metric(f"📝 Classe prédite :", f"{text_top_cat} - {text_top_class}")
                form_cols[2].metric("💪 Confiance", f"{text_confidence:.2f}%")
                
                st.success(f"📝 Classe prédite : {text_top_cat} - {text_top_class} - confiance : {text_confidence:.2f}%")
                st.bar_chart(text_proba, horizontal=True)
            
            if not text_input and  uploaded_file:

                # Afficher les résultats du modèle d'image
                form_cols[2].metric(f"📸 Classe prédite :", f"{top_cat} - {top_class}")
                form_cols[2].metric("💪 Confiance", f"{confidence:.2f}%")

                # Afficher les résultats
                st.success(f"📸 Classe prédite : {top_cat} - {top_class}  - confiance : {confidence:.2f}%")
                st.bar_chart(img_proba, horizontal=True)

                
                # Afficher le grad-CAM
                cols = st.columns(3)
                grad_cam_image, predicted_class, predicted_score = grad_cam(input_tensor, model, model.get_layer("top_conv"))
                # Afficher l'image originale
                cols[0].image(image, caption="Image chargée", width=224)
                # Afficher l'image prétraitée
                cols[1].image(cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB), caption="Image traitée", width=224)
                # Afficher l'image avec Grad-CAM
                cols[2].image(grad_cam_image, caption="Image avec Grad-CAM", width=224)

            if text_input and uploaded_file:    
                col1, spacer, col2 = st.columns([1, 0.1, 1])
                # Interprétabilité
                col1.success(f"📸 Classe prédite : {top_cat} - {top_class}  - confiance : {confidence:.2f}%")
                col1.bar_chart(img_proba, horizontal=True)
                col2.success(f"📝 Classe prédite : {text_top_cat} - {text_top_class} - confiance : {text_confidence:.2f}%")
                col2.bar_chart(text_proba, horizontal=True)
                

                # Afficher le grad-CAM
                cols = col1.columns([0.9, 0.1, 0.9, 0.1, 0.9])
                grad_cam_image, predicted_class, predicted_score = grad_cam(input_tensor, model, model.get_layer("top_conv"))
                # Afficher l'image originale
                cols[0].image(image, caption="Image chargée", width=224)
                # Afficher l'image prétraitée
                cols[2].image(cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB), caption="Image traitée", width=224)
                # Afficher l'image avec Grad-CAM
                cols[4].image(grad_cam_image, caption="Image avec Grad-CAM", width=224)


                

                # 1. Récupérer le vocabulaire
                feature_names = vectorizer.get_feature_names_out()

                # 2. Récupérer le vecteur TF-IDF (en supposant un seul texte)
                tfidf_vector = text_input_vectorized[0]

                # 3. Créer un DataFrame avec les mots et leurs scores
                df_tfidf = pd.DataFrame({
                    'mot': feature_names,
                    'score': tfidf_vector
                })

                # 4. Filtrer les mots non pertinents (score 0) et trier
                df_tfidf = df_tfidf[df_tfidf['score'] > 0].sort_values(by='score', ascending=False).reset_index(drop=True)

                col2.dataframe(df_tfidf.head(10), use_container_width=True)
                


