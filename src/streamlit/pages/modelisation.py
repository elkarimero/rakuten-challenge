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


text_models_simple, text_models_best, text_transfert, benchmark_models_images, model_efficientnet = st.tabs(["Données texte - Modèles simples",
                                                                                                             "Données texte - Modèle retenu",
                                                                                                             "Données texte - Transfert learning",
                                                                                                             "Images - Transfert learning",
                                                                                                             "🏆 Images - Modèle retenu"])


# ONGLET 1: DONNEES TEXTE - MODELES SIMPLES
with text_models_simple:
    st.header("Données texte - modèles simples")

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
    pred_path = f"data/Predictions/y_pred_{model_code}.npy"
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
      
      col1, col2 = st.columns(2)
      with col1:
        st.pyplot(fig, use_container_width=True)

      
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



# ONGLET 2: DONNEES TEXTE - MODELE RETENU
with text_models_best:
    st.header("Données texte - Modèle retenu")

    # Nom des modèles
    model_names = [
        "K-Nearest Neighbors",
        "Decision Tree",
        "Naive Bayes",
        "Logistic Regression",
        "Ridge Classifier",
        "Linear SVM"
        ]

    # Chargement des prédictions
    y_pred_knn = np.load("data/Predictions/y_pred_k-nearest_neighbors.npy")
    y_pred_tree = np.load("data/Predictions/y_pred_decision_tree.npy")
    y_pred_nb = np.load("data/Predictions/y_pred_naive_bayes.npy")
    y_pred_lr = np.load("data/Predictions/y_pred_logistic_regression.npy")
    y_pred_rdg = np.load("data/Predictions/y_pred_ridge_classifier.npy")
    y_pred_svm = np.load("data/Predictions/y_pred_linear_svm.npy")

    # Rapports
    report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
    report_tree = classification_report(y_test, y_pred_tree, output_dict=True)
    report_nb = classification_report(y_test, y_pred_nb, output_dict=True)
    report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
    report_rdg = classification_report(y_test, y_pred_rdg, output_dict=True)
    report_svm = classification_report(y_test, y_pred_svm, output_dict=True)

    # F1 scores
    f1_knn = report_knn["macro avg"]["f1-score"]
    f1_tree = report_tree["macro avg"]["f1-score"]
    f1_nb = report_nb["macro avg"]["f1-score"]
    f1_lr= report_lr["macro avg"]["f1-score"]
    f1_rdg = report_rdg["macro avg"]["f1-score"]
    f1_svm = report_svm["macro avg"]["f1-score"]

    f1_scores = [
        f1_knn,
        f1_tree,
        f1_nb,
        f1_lr,
        f1_rdg,
        f1_svm
        ]

    f1_df = pd.DataFrame({
        "Modèle": model_names,
        "F1_avg": f1_scores
    }).sort_values(by="F1_avg", ascending=False)


    # Graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=f1_df, x="F1_avg", y="Modèle", hue="Modèle", palette="viridis", dodge=False, legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', label_type='edge', padding=3)

    ax.set_title("Comparaison des performances des modèles (F1-score)")
    ax.set_xlabel("F1-score")
    ax.set_ylabel("Modèle")
    ax.set_xlim(0, 1.05)
    plt.tight_layout()

    col1, col2 = st.columns(2)
    with col1:
        st.write("Le modèle retenu est le Linear SVM avec C=1")
        st.pyplot(fig)



# ONGLET 3: DONNEES TEXTE - TRANSFERT LEARNING
with text_transfert:
    st.header("Données texte - Transfert learning")
 
    model_choice = st.selectbox("Choisir un modèle pré-entrainé:", ["BERT","ADAM"])
    
    # Résumé des stats
    if model_choice == "BERT":
        st.subheader("Modèle pré-entrainé de BERT")
        col1, col2, col3, col4, col5 = st.columns(5)
        col2.metric("Test Accuracy", "87%", "+6%")
        col3.metric("F1 score", "87%", "+7%")
        col4.metric("Entrainement", "9 heures", "vs 15-30 min", delta_color = "inverse")
        
    elif model_choice == "ADAM":
        st.subheader("Modèle pré-entrainé ADAM")
        col1, col2, col3, col4, col5 = st.columns(5)
        col2.metric("Test Accuracy", "79.43%", "-3%")
        col3.metric("F1 score", "80.22%", "-2%")

    # Graphiques
    col1, col2 = st.columns(2)
    if model_choice == "BERT":        
        with col1:
            st.image("Plots/BERT_learning_curve.png", caption="Learning Curve")
        with col2:
            st.image("Plots/BERT_F1_score_par_classe.png", caption="F1 scores")

    elif model_choice == "ADAM":
        
        with col1:
            st.image("Plots/ADAM_result.png", caption= "Result ADAM")



# ONGLET 4: IMAGES - TRANSFERT LEARNING
with benchmark_models_images:

    st.header("Images - Transfert learning")
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
        df = df.sort_values(by="F1 Score", ascending=False)

        # Définir les colonnes à optimiser
        max_cols = ["F1 Score"]
        min_cols = [] #["Test Loss", "Params", "Training Time (s)"]
        
        # Création du style avec gradient
        styled_df = df.style

        # Appliquer un gradient croissant pour les colonnes à maximiser
        styled_df = styled_df.background_gradient(subset=max_cols, cmap='Greens')

        # Appliquer un gradient inverse pour les colonnes à minimiser
        styled_df = styled_df.background_gradient(subset=min_cols, cmap='Greens_r')
        return styled_df
    
    style_df_base = load_data("./data/benchmark_results_base.csv")

    st.subheader("Benchmark des modèles")
    st.dataframe(style_df_base, use_container_width=True)



# ONGLET 5: IMAGES - MODELE RETENU
with model_efficientnet:
    st.subheader("🏆 EfficientNetB0")
    st.write("""
    Le modèle EfficientNetB0 a été sélectionné pour sa performance optimale en termes de précision et de F1 Score, tout en maintenant un nombre de paramètres raisonnable et un temps d'entraînement acceptable.
    """)

    with st.expander("**Performances du modèle avec et sans fine-tuning**", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Test Accuracy", "62,9%", "+12")
        col2.metric("Test Loss", "1.49", "-0,15", delta_color="inverse")
        col3.metric("F1 score", "62,8%", "+12")
        col4.metric("Paramètres", "4,4 millions", "4%", delta_color="inverse")
        col5.metric("Entrainement", "45 minutes", "-9 min", delta_color="inverse")


    with st.expander("**Comparaison avec ResNet50**"):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Test Accuracy", "62,9%", "+4,5")
        col2.metric("Test Loss", "1.49", "-0,08", delta_color="inverse")
        col3.metric("F1 score", "62,8%", "+4")
        col4.metric("Paramètres", "4,4 millions", "-20 millions", delta_color="inverse")
        col5.metric("Entrainement", "45 minutes", "-13 min", delta_color="inverse")
    
    with st.expander("**Interprétabilité**"):
        img_cols = st.columns([1,0.1,0.9])

        img_cols[0].image("./images/grad_cam.png", caption="Analyse Grad-CAM sur une image de test")
        #img_cols[1].image("./images/efficientnet_training.png", caption="Résultats du modèle EfficientNetB0 sur le dataset de test")

       # Attention plus focalisée
        img_cols[2].markdown("""
                             Le modèle fine-tuned montre des performances bien meilleures, avec 
                             - des activations plus précises, l'attention est plus **focalisée sur l'objet d'intérêt**
                             - des prédictions **plus confiantes**""")