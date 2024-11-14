import logging
from sklearn.naive_bayes import GaussianNB
from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_naive_bayes():
    logging.info("Loading and preprocessing data...")
    X, y = load_spotify_dataset()
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split(X, y)
    logging.info("Data loaded and split into training and validation sets.")

    model = GaussianNB()
    logging.info("Training Naive Bayes model...")
    model.fit(X_train, y_train)
    logging.info("Model training completed.")

    val_preds = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    logging.info(f"Naive Bayes Validation Accuracy: {val_accuracy:.4f}")

    return model, val_accuracy