import pandas as pd
import numpy as np
import seaborn as sns
import math
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

def evalute_models(model_dict,y_test,averages=None):
    score_list = []
    for model_name,y_pred in model_dict.items():
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=averages)
        recall = recall_score(y_test, y_pred, average = averages)
        f1 = f1_score(y_test, y_pred, average=averages)

        model_score = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
        }
        print(f"=== CLASSIFICATION REPORT: {model_name} ===")
        print(classification_report(y_test, y_pred))
        print("\n" + "="*40 + "\n") # Pembatas antar model

        score_list.append(model_score)
    df_final = pd.DataFrame(score_list)
    return df_final

def plot_confusion_matrix(model_dict,y_test,labels=[...]):
    fig, axes = plt.subplots(3,3,figsize=(10,10))
    axes = axes.flatten()
    for i, (model_name, y_pred) in enumerate(model_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False, ax=axes[i])
        axes[i].set_title(f"Confusion Matrix — {model_name}")
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")
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