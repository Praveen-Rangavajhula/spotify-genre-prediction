import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_logistic_regression():
    logging.info("Loading and preprocessing data...")
    X, y = load_spotify_dataset()
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split(X, y)
    logging.info("Data loaded and split into training and validation sets.")

    # Reduced search space
    param_dist = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga']
    }

    model = LogisticRegression(max_iter=1000, random_state=42, warm_start=True)

    # RandomizedSearchCV for faster tuning
    random_search = RandomizedSearchCV(
        model, param_distributions=param_dist, n_iter=4, scoring='accuracy', cv=3, n_jobs=-1, random_state=42
    )
    logging.info("Starting randomized search for hyperparameter tuning...")
    random_search.fit(X_train, y_train)
    logging.info("Randomized search completed.")

    best_model = random_search.best_estimator_
    logging.info(f"Best model parameters: {best_model.get_params()}")

    # Evaluate on validation set
    val_preds = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    logging.info(f"Best Logistic Regression Validation Accuracy: {val_accuracy:.4f}")

    return best_model, val_accuracy
