# utils.py
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

import seaborn as sns


# Directory to save results
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_metrics(model_name, y_test, y_pred, val_accuracy, test_accuracy, normalize=True, top_n_classes=10):
    import numpy as np

    # Save metrics as JSON
    metrics = {
        "Validation Accuracy": val_accuracy,
        "Test Accuracy": test_accuracy,
        "Classification Report": classification_report(y_test, y_pred, output_dict=True),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    with open(f"{RESULTS_DIR}/{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Prepare Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true' if normalize else None)

    # Get class labels
    class_labels = np.unique(y_test)

    # Handle Top-N Classes
    if top_n_classes:
        unique_classes, class_counts = np.unique(y_test, return_counts=True)
        sorted_indices = np.argsort(-class_counts)  # Sort by frequency
        top_classes = unique_classes[sorted_indices][:top_n_classes]  # Select Top-N

        # Map top_classes (labels) to indices
        top_class_indices = [np.where(class_labels == cls)[0][0] for cls in top_classes]

        # Slice confusion matrix
        cm = cm[np.ix_(top_class_indices, top_class_indices)]
        y_test_labels = top_classes
        y_pred_labels = top_classes
    else:
        y_test_labels = y_pred_labels = class_labels  # Use all labels

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=y_test_labels, yticklabels=y_pred_labels
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(f"{RESULTS_DIR}/{model_name}_confusion_matrix.png")
    plt.close()


def plot_roc_curve(models, X_test, y_test):
    """Plot ROC curves for all models in a multiclass setting."""
    # Binarize the output
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)

    plt.figure(figsize=(10, 8))
    for model_name, (model, _) in models.items():
        # Check if model supports predict_proba
        if not hasattr(model, "predict_proba"):
            continue  # Skip models that don't support probability predictions

        y_score = model.predict_proba(X_test)

        # Compute micro-average ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.title("ROC Curves for Models (Micro-average)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.savefig(f"{RESULTS_DIR}/roc_curves.png")
    plt.close()
