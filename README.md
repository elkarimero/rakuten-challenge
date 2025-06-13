Challenge Rakuten
==============================

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               
    │   ├── interim        <- Dataset partiellement traités (avant traduction automatique)
    │   ├── processed      <- Dataset finaux pour la modélisation
    │   └── raw            <- Le dump de données original et immuable
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Notebook de travail
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports                                     <- Rapport de projet et artefacts
    │   └── figures                                 <- Graphiques et statistiques générés et destinés à être utilisés dans les rapports
    │   │   ├── analyse_explo_images                <- artefacts de l'analyse exploratoires des images
    │   │   ├── benchmark_images_models_results     <- artefacts du benchmark des models pour les images
    │   │   ├── benchmark_images_models_results     <- artefacts du benchmark des models pour le text
    │   │   └── models_training                     <- Graphs d'apprentissage du modèle
    │
    ├── requirements.txt   <- fichier de requirements pour reproduire l'environnement du projet
    │
    ├── src                <- Code source du projet
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── image_features.py                   <- Méthodes utilisées pour le preprocessing des images
    │   │   ├── images_dataset_resampling.py        <- Méthodes utilisées pour le resampling des images
    │   │   ├── images_preprocessing.py             <- Pipeline de preprocessing des images
    │   │   └── translation_script.py               <- Script de traduction des descriptions produits
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    |   │   ├── text_model_pytorch                       
    |   │   │   ├── data_utils.py                   <- Contient les fonctions pour charger et prétraiter les données.
    |   │   │   ├── main.py                         <- Le script principal qui utilise les fonctions des autres fichiers pour exécuter le pipeline complet
    |   │   │   ├── evaluate.py                     <- Les fonctions pour évaluer les modèles et visualiser les résultats
    |   │   │   ├── models.py                       <- Définitions des modèles
    |   │   │   └── train.py                        <- Fonctions pour entraîner les modèles
    |   │   ├── images_models                       
    |   │   │   ├── dataset_utils.py                <- Méthodes utiles pour manipuler le dataset d'image
    |   │   │   ├── dataviz_utils.py                <- Méthodes utiles pour générer des graphiques sur l'entrainement du modèle
    |   │   │   └── EfficientNetBo_model_train.py   <- Script d'entrainement du modèle
    │   │   ├── benchmark_images_models.py          <- Script de benchmark des modèles pour les images
    │   │   └── predict_model.py                        
    │   │
    │   ├── visualization  
    |   │   ├── analyse_explo_images.py             <- Script de création de visualisation pour l'analyse explo des images 
    │   │   └── grad-cam.py                         <- visualisation grad-cam

--------
