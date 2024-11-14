from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score

def train_decision_tree():
    # Load and split the data
    X, y = load_spotify_dataset()
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split(X, y)

    # Define hyperparameter grid
    param_grid = {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    # Use GridSearchCV to find the best hyperparameters
    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    # Best model and its validation accuracy
    best_model = grid_search.best_estimator_
    val_preds = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)

    print(f"Best Decision Tree Validation Accuracy: {val_accuracy:.4f}")
    return best_model, val_accuracy
