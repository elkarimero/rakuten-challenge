# Pour visualiser les performances
import matplotlib.pyplot as plt
import os

def display_results(model_history, model_name, save_dir=None):
    """
    Affiche les résultats d'entraînement du modèle.

    Args:
        model_history: Historique de l'entraînement du modèle.
        model_name: Nom du modèle pour l'affichage.
    """
    # Récupérer les données d'entraînement et de validation
    train_loss = model_history.history["loss"]
    val_loss = model_history.history["val_loss"]
    train_accuracy =  model_history.history["accuracy"]
    val_accuracy = model_history.history["val_accuracy"]
    
    plt.figure(figsize=(20, 8))
    
    # Tracer la perte
    plt.subplot(121)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title(model_name + ": Perte d'entraînement et de validation")
    plt.ylabel('Perte ')
    plt.xlabel('Époque')
    plt.legend(['Entraînement', 'Validation'], loc='best')
    
    # Tracer l'erreur absolue moyenne (MAE)
    plt.subplot(122)
    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.title(model_name+': Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Époque')
    plt.legend(['Entraînement', 'Validation'], loc='best')

    plt.tight_layout()
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{model_name}_training_results.png")
        plt.savefig(filename)
        print(f"Graphiques sauvegardés dans : {filename}")
    
    plt.show()