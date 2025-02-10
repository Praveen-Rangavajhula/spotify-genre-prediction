import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from data.data_loader import load_spotify_dataset, train_val_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_random_forest():
    logging.info("Loading and preprocessing data...")
    X, y = load_spotify_dataset()
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split(X, y)
    logging.info("Data loaded and split into training and validation sets.")

    # Parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    model = RandomForestClassifier(random_state=42, warm_start=True)
    logging.info("Starting randomized search for hyperparameter tuning...")
    random_search = RandomizedSearchCV(
        model, param_distributions=param_dist, n_iter=5,
        scoring='accuracy', cv=3, n_jobs=4, random_state=42
    )
    random_search.fit(X_train, y_train)
    logging.info("Randomized search completed.")

    best_model = random_search.best_estimator_
    logging.info(f"Best model parameters: {best_model.get_params()}")

    val_preds = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    logging.info(f"Best Random Forest Validation Accuracy: {val_accuracy:.4f}")

    # Get feature importances
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("Top 10 Important Features:")
    print(feature_importances.head(10))

    # Plot the feature importances (adjust figure size for ICML)
    plt.figure(figsize=(3.25, 2.5))  # width ~3.25 inches for 2-column ICML format
    plt.barh(feature_importances['Feature'].head(10), feature_importances['Importance'].head(10))
    plt.gca().invert_yaxis()
    plt.xlabel('Importance', fontsize=8)
    plt.ylabel('Feature', fontsize=8)
    plt.title('Top 10 Feature Importances from Random Forests', fontsize=9)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('random_forest_feature_importances.png', dpi=300, bbox_inches='tight')

    return best_model, val_accuracy
