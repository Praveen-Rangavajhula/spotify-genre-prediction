import logging
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_knn():
    logging.info("Loading and preprocessing data...")
    X, y = load_spotify_dataset()

    # Convert data to NumPy arrays and ensure they are C-contiguous
    X_np = np.ascontiguousarray(X.to_numpy())
    y_np = y.to_numpy()

    # Split the data
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
    )
    logging.info("Data loaded and split into training and validation sets.")

    # Define hyperparameter grid for KNN
    param_dist = {
        'n_neighbors': [3, 5, 10, 20],  # Test with different numbers of neighbors
        'weights': ['uniform', 'distance'],  # Weighting schemes
        'metric': ['euclidean', 'manhattan']  # Distance metrics
    }

    # Initialize the KNN model
    model = KNeighborsClassifier()

    # Subsample data for hyperparameter tuning
    subsample_frac = 0.4  # Use 20% of the training data
    X_train_sub_np, _, y_train_sub_np, _ = train_test_split(
        X_train_np, y_train_np, train_size=subsample_frac, random_state=42, stratify=y_train_np
    )
    logging.info(f"Subsampled {len(X_train_sub_np)} training examples for hyperparameter tuning.")

    # Randomized search for hyperparameter tuning
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=5,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        random_state=42,
    )
    logging.info("Starting randomized search for hyperparameter tuning...")
    random_search.fit(X_train_sub_np, y_train_sub_np)
    logging.info("Randomized search completed.")

    # Get the best model
    best_model = random_search.best_estimator_
    logging.info(f"Best model parameters: {best_model.get_params()}")

    # Evaluate on validation set
    val_preds = best_model.predict(X_val_np)
    val_accuracy = accuracy_score(y_val_np, val_preds)
    logging.info(f"Best KNN Validation Accuracy: {val_accuracy:.4f}")

    return best_model, val_accuracy
