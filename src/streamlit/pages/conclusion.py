import streamlit as st

st.title("Conclusions et perspectives")


with st.expander("**Le bilan**", expanded=True):
    cols = st.columns([1,0.30,0.30])
    cols[0].markdown("""Nos objectifs sont globalement atteints puisque nous avons réussi à:
    """)
    cols[0].markdown("""
                - Améliorer la qualité des données (nettoyage, traduction, équilibrage …)
                - Exploiter efficacement les données textuelles et visuelles (en mettant en place des pipelines complet des deux côtés)
                - Classer automatiquement les produits Rakuten avec une bonne performance (*macro-F1 score > 80%*)
    """)

    cols[1].metric("📸 F1-score Modèle image", "62%",border=True)
    cols[2].metric("📝 F1-score Modèle texte", "82%",border=True)

with st.expander("**Ce que nous avons appris**"):
    st.markdown("""
    - *Manipuler des* **techniques avancées de NLP** *(traduction automatique, vectorisation TF-IDF et lemmatisation)*
    - *Différentes techniques de* **traitement d'images** *(data augmentation, détection d'objets et contours)*
    - *Utiliser et fine tuner des* **modèles pré-entraînés** 
    - Fusionner les données textuelles et visuelles pour améliorer la classification
    """)

with st.expander("**Les difficultés rencontrées**"):
    st.markdown("""
    - *Complexité du* **nettoyage de données textuelles** *(langues multiples, bruit HTML, caractères spéciaux, problème d'encoding)*
    - *La nécessité de gérer deux modalitées qui a demandé une* **montée en compétence rapide en début de projet** *(beaucoup d'avance de phase sur les cours)*
    - *Ambiguïté et l'hétérogénéité de certaines classes taxonomiques*
    - **Temps de calcul important** *pour l'entraînement des modèles, nécessitant une gestion efficace des ressources*
    """)


with st.expander("**Perspectives**"):
    st.markdown("""
    - *Utilisation de type transformer comme* **Vision Transformers (ViT)** *pour la partie image*
    - *Expérimenter d'autres technique de fusion des approches ou l’usage d’un* **modèle multimodal (ex : CLIP, ViLT)**
    - *Revue de la taxonomie produit : subdiviser les classes trop hétérogènes* 
    """)
