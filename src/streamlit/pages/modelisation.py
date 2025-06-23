import streamlit as st
import pandas as pd

st.title("Modélisations")
st.write("""
lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
""")


explo_text_tab, explo_image_tab = st.tabs(["Données textuelles", "Images"])

with explo_text_tab:
    st.header("Exploration des données textuelles")
    
with explo_image_tab:
    

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

    st.header("Benchmark des modèles de base")
    st.dataframe(style_df_base, use_container_width=True)

    st.header("Benchmark des modèles finetunés")
    st.dataframe(style_df_finetuned, use_container_width=True)