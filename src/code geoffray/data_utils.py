import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('french'))
    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", text)
    tokens = word_tokenize(text, language='french')
    lemmas = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(lemmas)

def vectorize_text(data, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(data)
    return X, vectorizer

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def encode_labels(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return y_encoded, label_encoder

def prepare_data(file_path):
    df = load_data(file_path)
    df['merged_lemmatized'] = df['merged'].apply(preprocess_text)
    X, vectorizer = vectorize_text(df['merged_lemmatized'])
    y_encoded, label_encoder = encode_labels(df['prdtypecode'])
    X_resampled, y_resampled = apply_smote(X, y_encoded)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer, label_encoder
