from data_utils import prepare_data
from models import ProductClassifier
from train import train_model
from evaluate import evaluate_model

def main():
    file_path = 'clean_dataset.csv'
    X_train, X_test, y_train, y_test, vectorizer, label_encoder = prepare_data(file_path)

    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    model = ProductClassifier(input_size, num_classes)
    trained_model = train_model(model, X_train, y_train)

    evaluate_model(trained_model, X_test, y_test, label_encoder)

if __name__ == "__main__":
    main()
