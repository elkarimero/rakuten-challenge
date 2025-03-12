# Importations pour la construiction du modèle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

# Importation pour la transformation sur les images
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Resizing

# Importation de l'utilitaire image_dataset_from_directory de Keras
from keras.utils import image_dataset_from_directory

img_size = (500, 500)  # Taille cible
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/mnt/c/Users/karim/rakuten/images/data_clean/images_deep/train",
    image_size=img_size,
    batch_size=batch_size,
    subset="training",
    validation_split=0.2,
    seed=42
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/mnt/c/Users/karim/rakuten/images/data_clean/images_deep/train",
    image_size=img_size,
    batch_size=batch_size,
    subset="validation",
    validation_split=0.2,
    seed=42
)


# Définition de l'entrée du modèle
inputs = Input(shape=(500, 500, 3))

x = Resizing(100, 100)(inputs)    # Redimensionner les images à 100x100 pixels
x = Rescaling(1./255)(x)        # Normalisation des pixels pour avoir des valeurs entre 0 et 1

# Ajout de la couche de convolution
#x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="valid")(x)

# Ajout de la couche de pooling pour réduire la taille des données
#x = MaxPooling2D(pool_size=(2, 2), padding='valid')(x)

# Ajout d'une couche de dropout pour éviter le surapprentissage
#x = Dropout(0.3)(x)

# Applatir les données pour les passer à la couche dense
x = Flatten()(x)

# SCouche dense pour faire la prédiction
outputs = Dense(1, activation="linear")(x) 

# Définir le modèle avec les entrées et sorties spécifiées
model = Model(inputs=inputs, outputs=outputs)


model.compile(
    loss="mse",
    optimizer="adam",
    metrics=["mean_absolute_error"]
)

model_history = model.fit(train_ds,
                          validation_data=val_ds,# données d'entraînement
                          epochs=20)      # proportion de l'échantillon de test