import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def train_logistic_regression():
    logging.info("Loading and preprocessing data...")
    X, y = load_spotify_dataset()
    logging.info("Data loaded.")

    X_train, y_train, X_val, y_val, _, _ = train_val_test_split(X, y)
    logging.info("Data split into training and validation sets.")

    # Drop original features used in interaction terms
    features_to_drop = ['danceability', 'valence', 'energy', 'acousticness']
    X_train = X_train.drop(columns=features_to_drop)
    X_val = X_val.drop(columns=features_to_drop)
    logging.info("Dropped original features used in interaction terms.")

    # Define hyperparameter search space
    param_dist = {
        'C': [0.1, 1],
        'penalty': ['l2'],
        'solver': ['saga'],
    }

    # Set verbose=1 in LogisticRegression to monitor progress
    model = LogisticRegression(
        max_iter=50,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )

    # RandomizedSearchCV with verbose=2 to get detailed output
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=1,
        scoring='accuracy',
        cv=3,
        n_jobs=4,
        verbose=2,
        random_state=42
    )
    logging.info("Starting randomized search for hyperparameter tuning...")
    random_search.fit(X_train, y_train)
    logging.info("Randomized search completed.")

    best_model = random_search.best_estimator_
    logging.info(f"Best model parameters: {best_model.get_params()}")

    # Evaluate on validation set
    logging.info("Evaluating model on validation set...")
    val_preds = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    logging.info(f"Best Logistic Regression Validation Accuracy: {val_accuracy:.4f}")

    return best_model, val_accuracy


if __name__ == "__main__":
    train_logistic_regression()
