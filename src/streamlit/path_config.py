"""
Configuration des chemins pour fonctionner en local et sur Streamlit Cloud
"""
import os
from pathlib import Path

# Déterminer le répertoire courant de l'application Streamlit
STREAMLIT_DIR = Path(__file__).parent.absolute()

def get_data_path(relative_path: str) -> str:
    """
    Retourne le chemin complet pour un fichier dans le dossier data
    Fonctionne en local et sur Streamlit Cloud
    """
    return str(STREAMLIT_DIR / "data" / relative_path)

def get_model_path(relative_path: str) -> str:
    """
    Retourne le chemin complet pour un fichier dans le dossier models
    Fonctionne en local et sur Streamlit Cloud
    """
    return str(STREAMLIT_DIR / "models" / relative_path)

def get_image_path(relative_path: str) -> str:
    """
    Retourne le chemin complet pour une image
    Fonctionne en local et sur Streamlit Cloud
    """
    return str(STREAMLIT_DIR / "images" / relative_path)

# Exemples de chemins clés
DATA_PATH = STREAMLIT_DIR / "data"
MODELS_PATH = STREAMLIT_DIR / "models"
IMAGES_PATH = STREAMLIT_DIR / "images"

# Vérifier que les répertoires existent
assert DATA_PATH.exists(), f"Dossier data non trouvé: {DATA_PATH}"
assert MODELS_PATH.exists(), f"Dossier models non trouvé: {MODELS_PATH}"
