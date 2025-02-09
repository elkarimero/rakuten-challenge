import pandas as pd

from unidecode import unidecode
import re  
import html

def clean_text(text):
    if isinstance(text, str):            # Si le texte est une chaîne de caractères,
        text = unidecode(text)             # Remplace les accents par des lettres sans accent
        text = text.lower()                # Met le texte en minuscule
        text = ' '.join(text.split())      # Remplace les espaces inutiles
        return text
    return text

# Fonction qui identifie tout ce qui est délimité par < >
def extraction_balises(texte):
    return re.findall(r"<[^>]+>", str(texte))  

def build_text_features(df):

    # SUPPRESSION DES DOUBLONS EN COLONNE
    # On remplace alors la description par une chaine de caractère vide pour préparer la concaténation
    df.loc[df["designation"] == df["description"], "description"] = ""

    # CONCATENATION DE LA COLONNE DESIGNATION ET DESCRIPTION
    df["description"] = df["description"].fillna("")                      # On remplace les NaN par une chaine vide pour que la concaténation fonctionne
    df["designation"] = df["designation"] + " - " + df["description"]     # On concatène tout dans "designation"
    df = df.drop(columns = "description", axis =1)                        # On peut alors supprimer la colonne "description"

    # NORMALISATION DU TEXTE
    df["designation"] = df['designation'].apply(clean_text)

    # SUPPRESSION DES DOUBLONS DE LIGNES
    df = df.drop_duplicates(subset=["designation"])                       # On supprime les doublons en ne gardant que la première occurance

    # SUPPRESSION DES DOUBLES CHEVRONS
    df["designation"] = df["designation"].str.replace("<<", "").str.replace(">>", "")

    # Création d'une liste d'éléments uniques
    balises = set()                                   
    df["designation"].apply(lambda x: balises.update(extraction_balises(x)))

    # Création d'un dataframe et export Excel pour meilleur visibilité
    df_balises = pd.DataFrame(list(balises), columns=["Balises"])

    # todo il manque un bout de code ici pour supprimer les balises détectées dans le texte

    # TRAITEMENT DES CARACTERES SPECIAUX HTML (type &amp, &quot, etc)
    df["designation"] = df["designation"].str.replace("& amp ", "&amp;", regex=False)
    df["designation"] = df["designation"].str.replace("& amp;", "&amp;", regex=False)
    df["designation"] = df["designation"].str.replace("& nbsp ", "&nbsp;", regex=False)
    df["designation"] = df["designation"].str.replace("& nbsp;", "&nbsp;", regex=False)
    df["designation"] = df["designation"].str.replace("& lt ", "&lt;", regex=False)
    df["designation"] = df["designation"].str.replace("& lt;", "&lt;", regex=False)
    df["designation"] = df["designation"].str.replace("& gt ", "&gt;", regex=False)
    df["designation"] = df["designation"].str.replace("& gt;", "&gt;", regex=False)
    df["designation"] = df["designation"].str.replace("& ntsc ", "&ntsc;", regex=False)
    df["designation"] = df["designation"].str.replace("& ntsc;", "&ntsc;", regex=False)

    df["designation"] = df["designation"].apply(html.unescape)