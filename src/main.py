from src.logistic_regression import train_logistic_regression
from src.decision_tree import train_decision_tree
from src.svm import train_svm
from src.neural_network import train_neural_network
from src.naive_bayes import train_naive_bayes
from src.random_forest import train_random_forest

from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score


def main():
    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_spotify_dataset()

    # Split the data to obtain only the test set
    # Training and validation splits are handled in each model's function
    _, _, _, _, X_test, y_test = train_val_test_split(X, y)

    # Step 2: Train each model and retrieve the best model and validation accuracy
    models = {
        # 'Decision Tree': train_decision_tree(),
        # 'Logistic Regression': train_logistic_regression(),
        # 'SVM': train_svm(),
        # 'Neural Network': train_neural_network(),
        # 'Naive Bayes': train_naive_bayes(),
        'Random Forest': train_random_forest(),
    }

    # Step 3: Evaluate each best model on the test set
    print("\nModel Comparison on Test Set:")
    for model_name, (model, val_accuracy) in models.items():
        test_preds = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_preds)
        print(f"{model_name} - Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
