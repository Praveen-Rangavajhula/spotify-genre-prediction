import logging
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_neural_network():
    logging.info("Loading and preprocessing data...")
    X, y = load_spotify_dataset()
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split(X, y)
    logging.info("Data loaded and split into training and validation sets.")

    # Reduced parameter grid
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],  # Focus on fewer architectures
        'activation': ['relu'],  # Relu is the most common and effective
        'solver': ['adam'],  # Adam is faster and usually sufficient
        'learning_rate': ['constant', 'adaptive']
    }

    model = MLPClassifier(max_iter=200, random_state=42, early_stopping=True)

    # Use RandomizedSearchCV for faster tuning
    random_search = RandomizedSearchCV(
        model, param_distributions=param_dist, n_iter=4, scoring='accuracy', cv=3, n_jobs=-1, random_state=42
    )
    logging.info("Starting randomized search for hyperparameter tuning...")

    # Subsample data for faster hyperparameter tuning
    X_train_sub = X_train.sample(frac=0.5, random_state=42)
    y_train_sub = y_train[X_train_sub.index]
    random_search.fit(X_train_sub, y_train_sub)

    logging.info("Randomized search completed.")

    best_model = random_search.best_estimator_
    logging.info(f"Best model parameters: {best_model.get_params()}")

    # Evaluate on validation set
    val_preds = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    logging.info(f"Best Neural Network Validation Accuracy: {val_accuracy:.4f}")

    return best_model, val_accuracy
