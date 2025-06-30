import streamlit as st
import pandas as pd

st.title("Modélisations")
st.write("""
lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
""")


explo_text_tab, benchmark_models_images, model_efficientnet = st.tabs(["Données textuelles", "Transfert learning", "Modèle retenu : EfficientNetB0"])

with explo_text_tab:
    st.header("Exploration des données textuelles")
    
with benchmark_models_images:

    st.header("Transfert learning")
    col1, spacer, col2= st.columns([1,0.2,1])
    col1.subheader("Pourquoi utiliser des modèles pré-entraînés sur ImageNet ?")
    col1.markdown("""
                * **Réduction des besoins en données** *(le modèle a déjà appris à détecter une grande variété de caractéristiques visuelles sur plus d’un million d’images)*
                * **Moins de puissance de calcul requise** *(entrainement d'une partie seulement des couches du modèle)*
                * **ImageNet est un banque d’images généraliste** *(objets courants, véhicules, outils ...)*
                * Permet de se concentrer sur la classification des produits plutôt que sur l'extraction des caractéristiques visuelles
                """)
    col2.image("./images/imagenet.png", caption="ImageNet, un dataset de référence pour l'entraînement de modèles de vision par ordinateur")

    def load_data(filepath):
        # Chargement des données
        df = pd.read_csv(filepath)
        df[['Test Accuracy', "F1 Score", "Test Loss"]] = df[['Test Accuracy', "F1 Score", "Test Loss"]].astype(float)
        df[["Params", "Training Time (s)"]] = df[["Params", "Training Time (s)"]].astype(int)

        # Définir les colonnes à optimiser
        max_cols = ["Test Accuracy", "F1 Score"]
        min_cols = ["Test Loss", "Params", "Training Time (s)"]
        
        # Création du style avec gradient
        styled_df = df.style

        # Appliquer un gradient croissant pour les colonnes à maximiser
        styled_df = styled_df.background_gradient(subset=max_cols, cmap='Greens')

        # Appliquer un gradient inverse pour les colonnes à minimiser
        styled_df = styled_df.background_gradient(subset=min_cols, cmap='Greens_r')
        return styled_df
    
    style_df_base = load_data("./data/benchmark_results_base.csv")
    style_df_finetuned = load_data("./data/benchmark_results_fine_tuning.csv")

    st.subheader("Benchmark des modèles de base")
    st.dataframe(style_df_base, use_container_width=True)

    st.header("Benchmark des modèles finetunés")
    st.dataframe(style_df_finetuned, use_container_width=True)

with model_efficientnet:
    st.subheader("🏆 Modèle retenu : EfficientNetB0")
    st.write("""
    Le modèle EfficientNetB0 a été sélectionné pour sa performance optimale en termes de précision et de F1 Score, tout en maintenant un nombre de paramètres raisonnable et un temps d'entraînement acceptable.
    """)

    st.subheader("Performances du modèle")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Test Accuracy", "62,9%", "+12")
    col2.metric("Test Loss", "1.49", "-0,15", delta_color="inverse")
    col3.metric("F1 score", "62,8%", "+12")
    col4.metric("Paramètres", "4,4 millions", "4%")
    col5.metric("Entrainement", "45 minutes", "-9 min", delta_color="inverse")
    
    st.subheader("Entrainement du modèle")
    st.image("./images/efficientnet_training.png", caption="Résultats du modèle EfficientNetB0 sur le dataset de test")