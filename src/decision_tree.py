import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_decision_tree():
    logging.info("Loading and preprocessing data...")
    X, y = load_spotify_dataset()
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split(X, y)
    logging.info("Data loaded and split into training and validation sets.")

    param_grid = {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
    logging.info("Starting grid search for hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    logging.info("Grid search completed.")

    best_model = grid_search.best_estimator_
    logging.info(f"Best model parameters: {best_model.get_params()}")

    val_preds = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    logging.info(f"Best Decision Tree Validation Accuracy: {val_accuracy:.4f}")

    return best_model, val_accuracy