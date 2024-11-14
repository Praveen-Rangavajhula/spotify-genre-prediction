from sklearn.naive_bayes import GaussianNB
from data.data_loader import load_spotify_dataset, train_val_test_split
from sklearn.metrics import accuracy_score

def train_naive_bayes():
    X, y = load_spotify_dataset()
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split(X, y)

    model = GaussianNB()
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)

    print(f"Naive Bayes Validation Accuracy: {val_accuracy:.4f}")
    return model, val_accuracy
