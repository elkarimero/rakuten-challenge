import streamlit as st
#🧪🔍

# Menu latéral
intro_page = st.Page("pages/introduction.py", title="Introduction", icon="📖")
problematique_page = st.Page("pages/problematique.py", title="Problématique et enjeux", icon="❓")
exploration_page = st.Page("pages/exploration_datasets.py", title="Présentation et Exploration des datasets", icon="📊")
preprocessing_page = st.Page("pages/preprocessing.py", title="Méthodologie et Prétraitement des données", icon="🔧" )
modelisation_page = st.Page("pages/modelisation.py", title="Modélisation", icon="📈")
demo_page = st.Page("pages/demo.py", title="Démo", icon="🚀")
conclusion_page = st.Page("pages/conclusion.py", title="Conclusion", icon="✅")


# Création de la navigation
pg = st.navigation([
    intro_page, 
    problematique_page,
    exploration_page,
    preprocessing_page,
    modelisation_page,
    demo_page,
    conclusion_page])

# Configuration de la page
st.set_page_config(page_title="Projet Data Science", layout="wide")
pg.run()