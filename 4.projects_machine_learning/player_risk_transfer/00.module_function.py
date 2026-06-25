import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score

def evalute(y_true,y_pred,model_name='model'):
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }
    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1 Score:  {metrics['F1 Score']:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """Plot a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Did Not Survive", "Survived"],
        yticklabels=["Did Not Survive", "Survived"],
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(model_dict,X_test,y_test):
    plt.figure(figsize=(8,6))
    for name,model in model_dict.items():
        if hasattr(model,'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        fpr,tpr,_=roc_curve(y_test,y_prob)
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"Cross-Validation Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores

def compare_models(results_list):
    df = pd.DataFrame(results_list)
    df = df.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    return df