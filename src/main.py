
from src.neural_network import train_neural_network
from src.naive_bayes import train_naive_bayes
from src.random_forest import train_random_forest
from src.svm import train_svm
from src.decision_tree import train_decision_tree
from src.knn import train_knn
from utils import save_metrics, plot_roc_curve
from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score

def main():
    print("Loading and preprocessing data...")
    X, y = load_spotify_dataset()

    # Split the data
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    models = {
        # 'Decision Tree': train_decision_tree(),
        # 'Logistic Regression': train_logistic_regression(),
        # 'KNN': train_knn(),
        # 'SVM': train_svm(),
        # 'Neural Network': train_neural_network(),
        # 'Naive Bayes': train_naive_bayes(),
        'Random Forest': train_random_forest(),
    }
    # Train each model

    # Evaluate and save results
    for model_name, (model, val_accuracy) in models.items():
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        # Save metrics and plots using utils
        save_metrics(model_name, y_test, y_pred, val_accuracy, test_accuracy)


if __name__ == "__main__":
    main()