import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Fonction principale pour le benchmark
def benchmark_classifiers(df, text_column='text', target_column='category_name',save_dir_path='benchmark_text_models_results'):
    """
    Benchmark plusieurs modèles, vectorisation, resampling pour la classification des produits
    
    Args:
        df: DataFrame contenant les données
        text_column: nom de la colonne contenant le texte
        target_column: nom de la colonne contenant les étiquettes cibles
        save_dir_path: chemin du répertoire pour sauvegarder les résultats
    
    Returns:
        Un DataFrame avec les résultats du benchmark
    """
    # Vérification de l'existence du répertoire de sauvegarde
    os.makedirs(save_dir_path, exist_ok=True)

    # Vérification des données
    print(f"Nombre total d'échantillons: {len(df)}")
    print(f"Distribution des classes:")
    class_counts = df[target_column].value_counts()
    print(class_counts)
    
    # Visualisation de la distribution des classes
    plt.figure(figsize=(15, 8))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Distribution des catégories de produits')
    plt.xlabel('Catégorie')
    plt.ylabel('Nombre de produits')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_path, 'distribution_categories.png'))
    plt.close()
    
    # Préparation des données
    X = df[text_column]
    y = df[target_column]
    
    # Split train/test stratifié pour conserver les proportions des classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Taille de l'ensemble d'entraînement: {len(X_train)}")
    print(f"Taille de l'ensemble de test: {len(X_test)}")
    
    # Définir les vectoriseurs de texte à tester
    vectorizers = {
        #'CountVectorizer': CountVectorizer(max_features=10000),
        'TfidfVectorizer': TfidfVectorizer(max_features=10000)
    }
    
    # Définir les classifieurs à tester
    classifiers = {
        #'MultinomialNB': MultinomialNB(),
        'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0),
        #'LinearSVC': LinearSVC(C=1.0, class_weight=None, max_iter=1000),
        'SVC': SVC(gamma=0.01, kernel='linear'),
        'RandomForest': RandomForestClassifier(n_estimators=100),
        #'GradientBoosting': GradientBoostingClassifier(n_estimators=100)
    }
    
    # Techniques de rééquilibrage à tester
    resampling_methods = {
        #'SMOTE': SMOTE(random_state=42),
        #'RandomUnderSampler': RandomUnderSampler(random_state=42),
        'class_weight': 'use_class_weight_balanced'  # Pour les classifieurs qui supportent class_weight
    }
    
    # Préparation pour stocker les résultats
    results = []
    
    # Validation croisée stratifiée
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Boucle sur les combinaisons de modèles, vectoriseurs et méthodes de rééquilibrage
    for vec_name, vectorizer in vectorizers.items():
        for clf_name, classifier in classifiers.items():
            for resampling_name, resampling in resampling_methods.items():
                if  resampling_name == 'class_weight' and clf_name == 'GradientBoosting':
                    # Ignorer les classifieurs qui ne supportent pas class_weight
                    print(f"Skip: {vec_name} + {clf_name} + {resampling_name}")
                else:
                    # Activation du class_weight pour les classifieurs qui le supportent
                    if resampling_name == 'class_weight' and clf_name != 'GradientBoosting':
                        classifier_copy = classifier.__class__(**{**classifier.get_params(), 'class_weight': 'balanced'})
                    else:
                        classifier_copy = classifier
                    
                    print(f"Évaluation: {vec_name} + {clf_name} + {resampling_name}")

                    # Construire le pipeline
                    if resampling_name == 'class_weight':
                        pipeline = ImbPipeline([
                            ('vectorizer', vectorizer),
                            ('classifier', classifier_copy)
                        ])
                    else:
                        pipeline = ImbPipeline([
                            ('vectorizer', vectorizer),
                            ('resampler', resampling),
                            ('classifier', classifier_copy)
                        ])
                    
                    # Mesurer le temps d'exécution
                    start_time = time.time()
                    
                    # Entraînement sur l'ensemble complet
                    pipeline.fit(X_train, y_train)
                    
                    # Prédiction sur l'ensemble de test
                    y_pred = pipeline.predict(X_test)
                    
                    # Temps d'exécution
                    train_time = time.time() - start_time
                    
                    # Métriques d'évaluation
                    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
                    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                    
                    # Rapport de classification détaillé pour le modèle
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    report_df.to_csv(os.path.join(save_dir_path, f'classification_report_{vec_name}_{clf_name}_{resampling_name}.csv'))
                    
                    
                    # Ajouter les résultats au rapport de benchmark
                    results.append({
                        'Vectoriseur': vec_name,
                        'Classifieur': clf_name,
                        'Méthode de rééquilibrage': resampling_name,
                        'Precision (macro)': precision_macro,
                        'Recall (macro)': recall_macro,
                        'F1 (macro)': f1_macro,
                        'Precision (weighted)': precision_weighted,
                        'Recall (weighted)': recall_weighted,
                        'F1 (weighted)': f1_weighted,
                        'Temps d\'entraînement (s)': train_time
                    })
                    
                    # Afficher les résultats actuels
                    print(f"F1 (macro): {f1_macro:.4f}, F1 (weighted): {f1_weighted:.4f}")
                    print(f"Temps d'entraînement: {train_time:.2f} secondes\n")
    
    # Transformer les résultats en DataFrame

    results_df = pd.DataFrame(results)

    # Trier par F1 macro (descendant)
    results_df = results_df.sort_values('F1 (macro)', ascending=False)
    
    # Sauvegarder les résultats
    results_df.to_csv(os.path.join(save_dir_path, 'benchmark_results.csv'), index=False)
    
    
    # Visualiser les résultats des meilleurs modèles
    plt.figure(figsize=(14, 8))
    top_models = results_df.head(10)
    model_names = [f"{row['Vectoriseur']}\n+{row['Classifieur']}\n+{row['Méthode de rééquilibrage']}" 
                  for _, row in top_models.iterrows()]
    
    # Créer un barplot pour les F1 scores
    bar_width = 0.35
    x = np.arange(len(model_names))
    
    plt.bar(x - bar_width/2, top_models['F1 (macro)'], bar_width, label='F1 macro', color='skyblue')
    plt.bar(x + bar_width/2, top_models['F1 (weighted)'], bar_width, label='F1 weighted', color='lightcoral')
    
    plt.xlabel('Modèle')
    plt.ylabel('Score F1')
    plt.title('Comparaison des meilleurs modèles')
    plt.xticks(x, model_names, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_path, 'top_models_comparison.png'))
    
    plt.close()
    
    return results_df

# Exemple d'utilisation
if __name__ == "__main__":

    # Définir le chemin du dossier où se trouve ce script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Chargement des données
    df_path = os.path.join(BASE_DIR, '../../data/processed/df_avec_categorie_part3_traduit.csv')
    df = pd.read_csv(df_path)

    # création de la colonne 'text' à partir de 'trad' ou 'truncated'
    df['text'] = df.apply(lambda row: row['trad'] if not pd.isna(row['trad']) else row['truncated'], axis=1)
    #df_clean = df[['prdtypecode', 'text']]
    df_clean = df[['category_name', 'text']]

    # Chemin du dossier de sauvegarde
    save_dir_path = os.path.normpath(os.path.join(BASE_DIR, '../../reports/benchmark_text_models_results'))
  
    # Exécuter le benchmark
    results = benchmark_classifiers(df_clean, save_dir_path=save_dir_path)


    # Afficher les 5 meilleurs modèles
    print("\nTop 5 des modèles:")
    print(results.head(5))

    # Meilleur modèle pour un entraînement final
    best_config = results.iloc[0]
    print(f"\nMeilleure configuration: {best_config['Vectoriseur']} + {best_config['Classifieur']} + {best_config['Méthode de rééquilibrage']}")

