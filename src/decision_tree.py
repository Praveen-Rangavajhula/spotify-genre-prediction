import logging

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
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
        'max_depth': [10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    model = DecisionTreeClassifier(random_state=42)

    # Use RandomizedSearchCV for faster tuning
    grid_search = RandomizedSearchCV(
        model, param_distributions=param_grid, n_iter=5, scoring='accuracy', cv=3, n_jobs=4, random_state=42
    )

    logging.info("Starting randomized search for hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    logging.info("Randomized search completed.")

    best_model = grid_search.best_estimator_
    logging.info(f"Best model parameters: {best_model.get_params()}")

    val_preds = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    logging.info(f"Best Decision Tree Validation Accuracy: {val_accuracy:.4f}")

    # Get feature importances
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    })

    # Sort features by importance
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    # Display the top features
    print("Top 10 Important Features:")
    print(feature_importances.head(10))

    # Plot the feature importances
    plt.figure(figsize=(14, 8))
    plt.barh(feature_importances['Feature'].head(10), feature_importances['Importance'].head(10))
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 10 Feature Importances from Decision Tree')
    plt.show()

    return best_model, val_accuracy
