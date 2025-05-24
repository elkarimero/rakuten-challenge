import numpy as np
import time
import pandas as pd

# Pour charger les modèles
import tensorflow as tf
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB3, EfficientNetB7,
    ResNet50, ResNet101,
    VGG16, VGG19
)

# Pour importer le datasets
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

# pour la serialisation des preprocessing (pour éviter les problèmes de désérialisation au chargement du modèle)
from keras.saving import register_keras_serializable

# Pour visualiser les performances
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Configuration de TensorFlow   
print("TensorFlow version:")
print(tf.__version__)
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
tf.keras.backend.clear_session()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# Chargement du dataset
#dir_name = "/mnt/c/Users/karim/rakuten/images/data_clean/images_deep/sample"

# dir_name = path du dataset équilibré par resampling
dir_name = "/mnt/c/Users/karim/rakuten/images/data_clean/images_deep/sample_balanced"
img_size = (224, 224)  # Taille cible
batch_size = 32
class_names = sorted(os.listdir(dir_name))
NB_CLASSES = len(class_names)

train_ds = image_dataset_from_directory(
    dir_name,
    image_size=img_size,
    batch_size=batch_size,
    subset="training",
    validation_split=0.2,
    seed=42
)

val_ds = image_dataset_from_directory(
    dir_name,
    image_size=img_size,
    batch_size=batch_size,
    subset="validation",
    validation_split=0.2,
    seed=42
)

test_ds = image_dataset_from_directory(
    dir_name,
    image_size=img_size,
    batch_size=batch_size,
    subset="validation",
    validation_split=0.2,
    seed=42
)

# couches personnalisées pour gérer le pré-traitement des images 
@register_keras_serializable()   
class VGG16Preprocess(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.keras.applications.vgg16.preprocess_input(inputs)

@register_keras_serializable()   
class VGG19Preprocess(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.keras.applications.vgg19.preprocess_input(inputs)

@register_keras_serializable()   
class RestNetPreprocess(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.keras.applications.resnet.preprocess_input(inputs)

@register_keras_serializable()   
class EfficientnetPreprocess(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EfficientnetPreprocess, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.keras.applications.efficientnet.preprocess_input(inputs)
    
# Fonction de construction du modèle
def build_keras_model(base_model_fn, model_name, preprocess = None, input_shape=(224, 224, 3), num_classes=27):
    """
    Args:
        base_model_fn: Fonction de base du modèle (par exemple, EfficientNetB0, ResNet50, etc.)
        model_name: Nom du modèle pour l'identification
        preprocess: Fonction de prétraitement spécifique au modèle
        input_shape: Forme d'entrée des images
        num_classes: Nombre de classes de classification

    Returns:
        Un modèle Keras avec les couches de classification ajoutées.
    """
    base_model = base_model_fn(
        include_top = False,
        weights ='imagenet',
        input_shape = input_shape,
        name = model_name
    )
    base_model.trainable = True  # on conserve les poids du modèle de base gelés 
    
    inputs = tf.keras.Input(shape=input_shape)

    # Augmentation des données
    x = tf.keras.layers.RandomFlip("horizontal")(inputs) # flip horizontal aléatoire
    x = tf.keras.layers.RandomRotation(0.2)(x)           # rotation aléatoire de +/- 20%  
    x = tf.keras.layers.RandomZoom(0.2)(x)               # applique un zoom aléatoire de +/- 20%
    x = tf.keras.layers.RandomContrast(0.2)(x)           # contraste aléatoire de +/- 20%

    # Preprocessing spécifique au modèle
    x = preprocess(x)

    # modèle de base
    x = base_model(x)

    # Couches de classification
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Global Average Pooling pour réduire la dimensionnalité
    x = tf.keras.layers.BatchNormalization()(x)  # Normalisation des activations

    # Couches entièrement connectées
    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Couche de sortie
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)


def train_keras_model(model, save_dir_path, nb_epochs=20, finetuning=None):

    # Vérification de l'existence du répertoire de sauvegarde
    os.makedirs(save_dir_path, exist_ok=True)

    nb_fine_tune_layers = 0 
    #fine tuning du model
    if finetuning is not None:
        # On dégèle le dernier bloc de convolution
        print(f"🔓 Déblocage du bloc : {finetuning}")
        for layer in model.layers:
            if finetuning in layer.name:
                layer.trainable = True
                nb_fine_tune_layers += 1
            else:
                layer.trainable = False
    
    start_time = time.time()
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=nb_epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=2, min_lr=1e-6),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(save_dir_path, f'{model.name}_model_best.keras'),
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    save_weights_only=False
                )
            ]
        )
    
        duration = time.time() - start_time
        test_loss, test_acc = model.evaluate(test_ds)
    
        # ⚠️ Prédictions et F1-score
        y_true = []
        y_pred = []
        for images, labels in test_ds:
            preds = model.predict(images)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))
    
        f1 = f1_score(y_true, y_pred, average='macro')
    
        return {
            "Test Accuracy": round(test_acc, 4),
            "Test Loss": round(test_loss, 4),
            "F1 Score": round(f1, 4),
            "Params": model.count_params(),
            "Training Time (s)": int(duration),
            "Nb Fine Tuned layers": nb_fine_tune_layers
        }
        
    except Exception as e:
        print(f"Error training {model.name}: {e}")

# Execution du benchmark
if __name__ == "__main__":

    # Définir le chemin du dossier où se trouve ce script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Chemin du dossier de sauvegarde
    save_dir_path = os.path.normpath(os.path.join(BASE_DIR, '../../reports/benchmark_images_models_results2'))

    # Liste des modèles à benchmarker
    models_to_test = {
        #"EfficientNetB0": lambda: build_keras_model(EfficientNetB0, model_name = "EfficientNetB0", num_classes=NB_CLASSES, preprocess = EfficientnetPreprocess() ),
        #"EfficientNetB3": lambda: build_keras_model(EfficientNetB3, model_name = "EfficientNetB3", num_classes=NB_CLASSES, preprocess = EfficientnetPreprocess()),
        #"EfficientNetB7": lambda: build_keras_model(EfficientNetB7, model_name = "EfficientNetB7", num_classes=NB_CLASSES, preprocess = EfficientnetPreprocess()),
        #"ResNet50": lambda: build_keras_model(ResNet50, model_name = "ResNet50", num_classes=NB_CLASSES, preprocess = RestNetPreprocess()),
        "ResNet101": lambda: build_keras_model(ResNet101, model_name = "ResNet101", num_classes=NB_CLASSES, preprocess = RestNetPreprocess()),
        #"VGG16": lambda: build_keras_model(VGG16, model_name = "VGG16", num_classes=NB_CLASSES, preprocess = VGG16Preprocess()),
        "VGG19": lambda: build_keras_model(VGG19, model_name = "VGG19", num_classes=NB_CLASSES, preprocess = VGG19Preprocess())
    }

    # Boucle de benchmark
    results = {}
    results_fine_tuned = {}

    for model_name, builder in models_to_test.items():
        print(f"\n🔄 Training model: {model_name}")
        model = builder() 
        model._name = model_name
        
        finetuning = None
        if "VGG" in model_name:
            finetuning = "block5_"
        elif "ResNet" in model_name:
            finetuning = "conv5_"
        elif 'EfficientNet' in model_name:
            finetuning = "block7a"
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        results[model_name] = train_keras_model(model, save_dir_path = save_dir_path)
        results_fine_tuned[model_name] = train_keras_model(model, nb_epochs = 30,save_dir_path = save_dir_path, finetuning = finetuning)


    # Résumé final
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results = df_results.sort_values(by="Test Accuracy", ascending=False)
    print("\n📊 Benchmark Results:")
    print(df_results)
    df_results.to_csv(os.path.join(save_dir_path, f'benchmark_results_base.csv'))


    plt.figure(figsize=(10, 6))
    sns.barplot(x=df_results.index, y=df_results["F1 Score"])
    plt.xticks(rotation=45)
    plt.title("F1 Score par modèle (base)")
    plt.ylabel("F1 Score")
    plt.xlabel("Modèle")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_path, f'f1_scores_base_model_plot.png'))

    plt.show()

    # résultats fine tuned
    df_results = pd.DataFrame.from_dict(results_fine_tuned, orient='index')
    df_results = df_results.sort_values(by="Test Accuracy", ascending=False)
    print("\n📊 Benchmark Results:")
    print(df_results)
    df_results.to_csv(os.path.join(save_dir_path, f'benchmark_results_fine_tuning.csv'))


    plt.figure(figsize=(10, 6))
    sns.barplot(x=df_results.index, y=df_results["F1 Score"])
    plt.xticks(rotation=45)
    plt.title("F1 Score par modèle (fine tuned)")
    plt.ylabel("F1 Score")
    plt.xlabel("Modèle")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_path, f'f1_scores_fine_tuned_model_plot.png'))
    plt.show()
