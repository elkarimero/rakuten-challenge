from dataset_utils import *
from dataviz_utils import *

import tensorflow as tf 
from keras.saving import register_keras_serializable

# Définir le chemin du dossier où se trouve ce script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Chemin du dossier de sauvegarde
save_dir_path = os.path.normpath(os.path.join(BASE_DIR, '../../../models/EfficientNetB0/'))

# 1. charge le dataset d'images à partir du répertoire
dir_name = "/mnt/c/Users/karim/rakuten/images/data_clean/images_deep/sample_balanced"
#dir_name = "/mnt/c/Users/karim/rakuten/images/data_clean/images_deep/sample"
train_ds, val_ds = build_dataset_from_directory(dir_name, img_size=(224, 224), batch_size=64)

# 2. Prétraitement des dataset puor l'entrainement et la validation

# prétraitement spécifique au modèle EfficientNet
preprocess_fn = tf.keras.applications.efficientnet.preprocess_input 

# couche d'augmentation des données pour l'entraînement
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),   # flip horizontal aléatoire
    tf.keras.layers.RandomRotation(0.2),        # rotation aléatoire de +/- 20%
    tf.keras.layers.RandomZoom(0.2),            # zoom aléatoire de +/- 20%
    tf.keras.layers.RandomContrast(0.2),        # contraste aléatoire de +/- 20%
])

train_ds = train_ds.map(lambda x, y: preprocess_ds(x, y, preprocess_fn=preprocess_fn))
val_ds = val_ds.map(lambda x, y: preprocess_ds(x, y, preprocess_fn=preprocess_fn)) # pas d'augmentation pour la validation

class_names = sorted(os.listdir(dir_name))
nb_class = len(class_names) #nombre de classes

#train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)
#val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

# 3. Construction du modèle de classification d'images
        
# Modèle de base EfficientNetB0
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Geler d'abord le modèle de base
base_model.trainable = False 

# Couches de classification 
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(nb_class, activation='softmax')(x)

# Construction finale
model = tf.keras.Model(base_model.input, outputs)

# Execution du benchmark
if __name__ == "__main__":
    # 4. Compilation du modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. Entraînement du modèle
    model_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,  
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir_path, 'EfficientNetB0_model_best.weights.h5'), save_best_only=True, monitor='val_accuracy')
        ]
    )
    # 6. Sauvegarde du modèle
    model.save(os.path.join(save_dir_path, 'EfficientNetB0_model.keras'))
    # 7. Affichage de l'historique d'entraînement
    display_results(model_history, "EfficientNetB0 (sans augmentation ni fine tuning)", save_dir=save_dir_path)


