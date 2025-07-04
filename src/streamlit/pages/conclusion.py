import streamlit as st

st.title("Conclusions et perspectives")


with st.expander("**Le bilan**", expanded=True):
    st.markdown("""Nos objectifs sont globalement atteints puisque nous avons réussi à:
    """)
    st.markdown("""
                - Améliorer la qualité des données (nettoyage, traduction, équilibrage …)
                - Exploiter efficacement les données textuelles et visuelles (en mettant en place des pipelines complet des deux côtés)
                - Classer automatiquement les produits Rakuten avec une bonne performance (macro-F1 score > 80%)
    """)

with st.expander("**Ce que nous avons appris**", expanded=True):
    st.markdown("""
    - Manipuler des techniques avancées de NLP (traduction automatique, vectorisation TF-IDF et lemmatisation)
    - Différentes techniques de traitement d'image (data augmentation, détection d'objets)
    - Utiliser et fine tuner des modèles pré-entraînés 
    - Fusionner les données textuelles et visuelles pour améliorer la classification
    """)

with st.expander("**Les difficultés rencontrées**", expanded=True):
    st.markdown("""
    - Complexité du nettoyage textuel (langues multiples, bruit HTML)
    - Ambiguïté de certaines classes taxonomiques (notamment la classe "jeux société enfants")
    - La nécessité de gérer deux modalitées qui a demandé une montée en compétence sur un large spectre de modèles et de techniques spécifiques
    - Temps de calcul important pour l'entraînement des modèles, nécessitant une gestion efficace des ressources
    - Fusion des données textuelles et visuelles : montée en compétence sur les modèles de fusion
    """)


with st.expander("**Perspectives**", expanded=True):
    st.markdown("""
    - Utilisation de type transformer comme Vision Transformers (ViT) pour la partie image
    - Revue de la taxonomie produit : subdiviser les classes trop hétérogènes comme "Jeux société enfants"
    - Techniques de data augmentation plus ciblées par classe
    - Expérimenter d'autres technique de fusion des approches ou l’usage d’un modèle multimodal (ex : CLIP, ViLT)
    """)
