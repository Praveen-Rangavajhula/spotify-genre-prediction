from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score

def train_logistic_regression():
    X, y = load_spotify_dataset()
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split(X, y)

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'multi_class': ['multinomial'],
        'solver': ['lbfgs', 'saga']
    }

    model = LogisticRegression(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    val_preds = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)

    print(f"Best Logistic Regression Validation Accuracy: {val_accuracy:.4f}")
    return best_model, val_accuracy
