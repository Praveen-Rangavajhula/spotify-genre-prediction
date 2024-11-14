from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score

def train_neural_network():
    X, y = load_spotify_dataset()
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split(X, y)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'learning_rate': ['constant', 'adaptive']
    }

    model = MLPClassifier(max_iter=300, random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    val_preds = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)

    print(f"Best Neural Network Validation Accuracy: {val_accuracy:.4f}")
    return best_model, val_accuracy
