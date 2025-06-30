import streamlit as st

st.title("Problématique et enjeux")
st.write("""
La classification automatique des produits est un enjeu majeur pour les entreprises de vente en ligne, car elle permet d'améliorer l'expérience utilisateur en facilitant la recherche de produits et en optimisant la gestion des stocks. Cependant, cette tâche est complexe en raison de la diversité des produits, des variations dans les descriptions et des images associées.

""")

col1, col2 = st.columns(2)

col1.header("Enjeux économiques")

col1.markdown("Risques d'une mauvaise classification des produits :")
col1.markdown("- **Perte de ventes** *en raison d'une mauvaise visibilité des produits*")
col1.markdown("- **Augmentation des coûts** *de mise en ligne et de gestion des stocks*")
col1.markdown("- **Impact négatif sur l'expérience utilisateur** *et la fidélisation des clients*")


col1.markdown("****")

col1.markdown("Benefices d'une classification automatisée avec du machine learning:")
col1.markdown("- **Accélérer** *le temps de mise en ligne des produits*")
col1.markdown("- **Réduire les coûts liés** *à la classification manuelle*")
col1.markdown("- **Améliorer la précision** *de la classification des produits*")
col1.markdown("- **Expérience utilisateur améliorée** *grâce à une recherche plus efficace et des recommendations plus pertinentes*")


col2.header("Enjeux techniques")
col2.markdown("Défis techniques de la classification automatique des produits :")
col2.markdown("- **Diversité des données** à analyser : descriptions textuelles, images")
col2.markdown("- **Qualité des données** : scrapping de site web, multilingue, déséquilibre des classes")
col2.markdown("- **Besoin de ressources** lié à la taille du dataset d'images à analyser")

col2.markdown("****")

col2.markdown("Compétences mobilisées pour adresser ces défis :")
col2.markdown("- **Différentes techniques NLP** comme tokenisation, lemmatisation, filtrage des stop words et vectorisation")
col2.markdown("- **Rééquilibrage des datasets** via des méthode de resampling et de data augmentation pour gérer les déséquilibres de classes")
col2.markdown("- **Utilisation de techniques avancées** comme le fine-tuning de modèles pré-entraînés (transfert learning)")
