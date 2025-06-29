import streamlit as st

st.title("Problématique et enjeux")
st.write("""
La classification automatique des produits est un enjeu majeur pour les entreprises de vente en ligne, car elle permet d'améliorer l'expérience utilisateur en facilitant la recherche de produits et en optimisant la gestion des stocks. Cependant, cette tâche est complexe en raison de la diversité des produits, des variations dans les descriptions et des images associées.

""")

st.header("Enjeux économiques")

st.markdown("Risques d'une mauvaise classification des produits :")
st.markdown("- **perte de ventes** en raison d'une mauvaise visibilité des produits")
st.markdown("- **augmentation des coûts** de mise en ligne et de gestion des stocks en raison d'une mauvaise classification")
st.markdown("- **impact négatif sur l'expérience utilisateur** et la fidélisation des clients")

st.markdown("Benefices d'une classification automatisée avec du machine learning:")
st.markdown("- **accélérer** le temps de mise en ligne des produits")
st.markdown("- **réduire les liés** à la classification manuelle")
st.markdown("- améliorer la **précision** de la classification des produits")
st.markdown("- **expérience utilisateur améliorée** grâce à une recherche de produits plus efficace et de recommendations plus pertinentes")


st.header("Enjeux techniques")
st.markdown("Défis techniques de la classification automatique des produits :")
st.markdown("- **diversité des données** à analyser : descriptions textuelles, images, etc.")
st.markdown("- **qualité des données** : catalogue multilingue, qualité des descriptions obtenues par scrapping de site web, doublons, etc.")
st.markdown("- **besoin de ressources** lié à la taille du dataset d'images à analyser")

st.markdown("Compétences mobilisées pour adresser ces défis :")
st.markdown("- **différentes techniques NLP** comme tokenisation, lemmatisation, filtrage des stop words et vectorisation")
st.markdown("- **rééquilibrage des datasets** via des méthode de resampling et de data augmentation pour gérer les déséquilibres de classes")
st.markdown("- **utilisation de techniques avancées** comme le fine-tuning de modèles pré-entraînés (transfert learning)")
