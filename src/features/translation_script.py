import deepl
import pandas as pd
import time

def traduire_par_lots(fichier_entree, fichier_sortie, taille_lot=100, delai_entre_lots=2):
    """
    Traduit les entrées en anglais sans traduction par lots.
    
    Args:
        fichier_entree: Chemin vers le fichier CSV d'entrée
        fichier_sortie: Chemin vers le fichier CSV de sortie
        taille_lot: Nombre d'éléments à traduire par lot
        delai_entre_lots: Délai en secondes entre chaque lot pour éviter les limitations d'API
    """
    # Configuration de l'API DeepL
    auth_key = "7f03f88a-3bd9-4fd4-bbd1-c95a5574f3bd:fx"  # clé principale
    # auth_key = "3d4f32b5-9c3b-4ebf-8f7c-3ee448b618ff:fx"  # clé backup
    translator = deepl.Translator(auth_key)
    
    # Chargement du fichier
    df = pd.read_csv(fichier_entree)
    
    # Créer un masque pour les éléments à traduire
    mask = (df["langue"] == "en") & (df["trad"].isna())
    
    # Obtenir tous les indices des éléments à traduire
    indices_a_traduire = df[mask].index.tolist()
    
    print(f"Nombre total d'éléments à traduire : {len(indices_a_traduire)}")
    
    # Traitement par lots
    for i in range(0, len(indices_a_traduire), taille_lot):
        # Extraire les indices du lot actuel
        indices_lot = indices_a_traduire[i:i+taille_lot]
        
        print(f"Traduction du lot {i//taille_lot + 1}: éléments {i+1} à {min(i+taille_lot, len(indices_a_traduire))}")
        
        # Pour chaque indice du lot
        for idx in indices_lot:
            try:
                # Traduire le texte
                texte_original = df.loc[idx, "truncated"]
                if pd.notna(texte_original) and texte_original.strip():  # Vérifier que le texte n'est pas vide
                    resultat_traduction = translator.translate_text(texte_original, target_lang="FR")
                    # Stocker la traduction
                    df.loc[idx, "trad"] = str(resultat_traduction)
                    # Afficher un point pour montrer la progression
                    print(".", end="", flush=True)
                else:
                    print("x", end="", flush=True)  # Marquer les textes vides
            except Exception as e:
                print(f"\nErreur lors de la traduction de l'élément {idx}: {e}")
                # Enregistrer l'état actuel en cas d'erreur
                df.to_csv(f"../../notebooks/{fichier_sortie}_sauvegarde_urgence.csv", index=False)
        
        print(f"\nLot {i//taille_lot + 1} terminé. Sauvegarde intermédiaire...")
        # Sauvegarde intermédiaire après chaque lot
        df.to_csv(f"{fichier_sortie}_progress.csv", index=False)
        
        # Attendre entre les lots pour respecter les limites de l'API
        if i + taille_lot < len(indices_a_traduire):
            print(f"Pause de {delai_entre_lots} secondes avant le prochain lot...")
            time.sleep(delai_entre_lots)
    
    # Vérifier le nombre d'éléments restants à traduire
    elements_restants = len(df[mask])
    print(f"\nTraduction terminée! Nombre d'éléments restant à traduire : {elements_restants}")
    
    # Sauvegarder le fichier final
    df.to_csv(fichier_sortie, index=False)
    print(f"Fichier sauvegardé sous {fichier_sortie}")


# Exécuter la fonction
if __name__ == "__main__":

    traduire_par_lots(
            fichier_entree="../../data/interim/df_avec_categorie_part1.csv",
            fichier_sortie="../../data/processed/df_avec_categorie_part1_traduit.csv",
            taille_lot=100,
            delai_entre_lots=5
        )
    
    traduire_par_lots(
            fichier_entree="../../data/interim/df_avec_categorie_part2.csv",
            fichier_sortie="../../data/processed/df_avec_categorie_part2_traduit.csv",
            taille_lot=100,
            delai_entre_lots=5
        )