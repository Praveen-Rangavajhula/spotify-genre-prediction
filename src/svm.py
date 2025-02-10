import logging
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_svm():
    logging.info("Loading and preprocessing data...")
    X, y = load_spotify_dataset()
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split(X, y)
    logging.info("Data loaded and split into training and validation sets.")

    # Define a reduced parameter grid for faster tuning
    param_dist = {
        'C': [1, 10, 20],  # Regularization strength
    }

    # Using LinearSVC for faster training with linear kernel
    model = LinearSVC(random_state=42, max_iter=1000, dual=False)

    # Use RandomizedSearchCV for faster tuning
    random_search = RandomizedSearchCV(
        model, param_distributions=param_dist, n_iter=3, scoring='accuracy', cv=3, n_jobs=-1, random_state=42
    )
    logging.info("Starting randomized search for hyperparameter tuning...")

    # Subsample data for faster hyperparameter tuning
    subsample_frac = 0.2  # Use 20% of the data for faster training
    X_train_sub = X_train.sample(frac=subsample_frac, random_state=42)
    y_train_sub = y_train[X_train_sub.index]
    logging.info(f"Subsampled {len(X_train_sub)} training examples for hyperparameter tuning.")

    random_search.fit(X_train_sub, y_train_sub)

    logging.info("Randomized search completed.")

    best_model = random_search.best_estimator_
    logging.info(f"Best model parameters: {best_model.get_params()}")

    # Evaluate on validation set
    val_preds = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    logging.info(f"Best SVM Validation Accuracy: {val_accuracy:.4f}")

    return best_model, val_accuracy

