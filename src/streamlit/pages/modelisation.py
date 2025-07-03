import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix



st.title("Modélisations")
st.write("""
lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
""")


text_models_simple, text_transfert, benchmark_models_images, model_efficientnet = st.tabs(["Données textuelles - Modèles simples", "Données textuelles - Transfert learning", "Transfert learning", "Modèle retenu : EfficientNetB0"])



with text_models_simple:
    st.header("Données textuelles - modèles simples")

    # 1. Chargement des données
    y_train = joblib.load("data/y_train_final.pkl")
    y_test = joblib.load("data/y_test.pkl")

    # 2. Liste des modèles
    model_names = {
      "KNN": "k-nearest_neighbors",
      "Decision Tree": "decision_tree",
      "Naive Bayes": "naive_bayes",
      "Logistic Regression": "logistic_regression",
      "Ridge Classifier": "ridge_classifier",
      "Linear SVM": "linear_svm"
    }

    # 3. Variantes des modèles
    learning_curve_options = {
      "KNN": {
          "code": "k-nearest_neighbors",
          "param_variants": ["F1_vs_k", "k=2", "k=5", "k=10"]
      },
      "Decision Tree": {
          "code": "decision_tree",
          "param_variants": ["max_depth=10", "max_depth=50", "max_depth=None"]
      },
      "Naive Bayes": {
          "code": "naive_bayes",
          "param_variants": ["alpha=0.01", "alpha=0.1", "alpha=1", "alpha=2"]
      },
      "Logistic Regression": {
          "code": "logistic_regression",
          "param_variants": ["C=0.01", "C=0.1", "C=1", "C=10", "C=100"]
      },
      "Ridge Classifier": {
          "code": "ridge_classifier",
          "param_variants": ["alpha=0.01", "alpha=0.1", "alpha=1", "alpha=10", "alpha=100"]
      },
      "Linear SVM": {
          "code": "linear_svm",
          "param_variants": ["C=0.01", "C=0.1", "C=1", "C=10", "C=100"]
      }
    }
    
    # 4. Menu déroulant des modèles
    model_select = st.selectbox("Choisir un modèle :", list(model_names.keys()))
    model_code = model_names[model_select]

    # 5. Boutons radio 
    col1, col2 = st.columns(2)
    
    with col1:
      options = ["Courbe d'apprentissage", "Rapport de classification", "Matrice de confusion"]
      if model_select == "Logistic Regression":
        options.append("Importance des mots")
      view_option = st.radio("Affichage :", options)

    with col2:
      if view_option == "Courbe d'apprentissage" and model_select in learning_curve_options:
        params = learning_curve_options[model_select]["param_variants"]
        param_choice = st.radio("Paramètre :", params)
      else:
        param_choice = None

    # 6. Chargement des prédictions pour la matrice de confusion
    pred_path = f"Data/Predictions/y_pred_{model_code}.npy"
    y_pred = np.load(pred_path)
        
    # 7. Affichage
    if view_option == "Rapport de classification":
      report_dict = classification_report(y_test, y_pred, output_dict=True)
      report_df = pd.DataFrame(report_dict).transpose().round(2)
      st.subheader("Rapport de classification")
      st.dataframe(report_df)

    elif view_option == "Matrice de confusion":
      cm = confusion_matrix(y_test, y_pred)
      labels = np.unique(y_train)
      fig, ax = plt.subplots(figsize=(12, 9))
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                  xticklabels=labels, yticklabels=labels, ax=ax)
      ax.set_xlabel("Prédit")
      ax.set_ylabel("Réel")
      ax.set_title(f"Matrice de confusion - {model_select}")
      st.pyplot(fig, use_container_width=False)
      
    elif view_option == "Courbe d'apprentissage":
      if param_choice:
        safe_param = param_choice.replace("=", "-")
        variant_code = learning_curve_options[model_select]["code"]
        variant_path = f"Plots/learning_curve_{variant_code}_{safe_param}.png"
        st.image(variant_path, caption=f"{model_select} - {param_choice}")

    elif view_option == "Importance des mots":
      st.subheader("Importance des mots")
      col1, col2, col3 = st.columns(3)
      with col1:
          st.image("Plots/logistic_regression_importance_mots1.png", caption="Classe 10")
      with col2:
          st.image("Plots/logistic_regression_importance_mots2.png", caption="Classe 2403")
      with col3:
          st.image("Plots/logistic_regression_importance_mots3.png", caption="Classe 2705")


with text_transfert:
    st.header("Données textuelles - Transfert learning")
 
    model_choice = st.selectbox("Choisir un modèle pré-entrainé:", ["BERT","2"])
    
    # Résumé des stats
    if model_choice == "BERT":
        st.subheader("Modèle pré-entrainé de BERT")
        col1, col2, col3, col4, col5 = st.columns(5)
        col2.metric("Test Accuracy", "87%", "+6%")
        col3.metric("F1 score", "87%", "+7%")
        col4.metric("Entrainement", "9 heures", "vs 15-30 min", delta_color = "inverse")
        
    elif model_choice == "2":
        st.subheader("Modèle pré-entrainé 2")

    # Graphiques
    col1, col2 = st.columns(2)
    if model_choice == "BERT":        
        with col1:
            st.image("Plots/BERT_learning_curve.png", caption="Learning Curve")
        with col2:
            st.image("Plots/BERT_F1_score_par_classe.png", caption="F1 scores")

    elif model_choice == "2":
        
        with col1:
            st.image()
        with col2:
            st.image()



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