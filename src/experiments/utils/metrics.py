from typing import List
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def get_metrics(predictions: List[float], labels: List[float]) -> dict:
    """Get a selection of performance metrics.

    Args:
        predictions (List[float]): List of predictions.
        labels (List[float]): List of labels.

    Returns:
        dict: Dictionary of performance metrics.
    """
    
    # Gather performance metrics
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    roc_auc = roc_auc_score(labels, predictions)
    
    # Create dictionary of metrics
    metrics = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp+tn)/(tp+fp+fn+tn),
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp),
        "f1_score": fscore,
        "roc_auc": roc_auc
    }

    return metrics


def print_metrics(metrics: dict):

    # Print metrics
    print("-"*5, f"Performance metrics", "-"*5)
    print(f"\nConfusion matrix")
    print(f"\"The number of True Positives, True Negatives, False Positives and False Negatives.\"")
    print(f"TP: {metrics['tp']} | TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']}")

    print(f"\nAccuracy ((TP + TN) / (TP + FP + FN + TN))")
    print(f"\"The percentage of all classifications which is true.\"")
    print(f"{round(metrics['accuracy']*100, 3)}%")

    print(f"\nPrecision (TP / (TP + FP))")
    print(f"\"The percentage of classified positives which is true.\"")
    print(f"{round(metrics['precision']*100, 3)}%")
    
    print(f"\nRecall (TP / (TP + FN))")
    print(f"\"Out of all the actual positives, how many did we correctly classify?\"")
    print(f"{round(metrics['recall']*100, 3)}%")

    print(f"\nSpecificity (TN / (TN + FP))")
    print(f"\"Out of all the actual negatives, how many did we correctly classify?\"")
    print(f"{round(metrics['specificity']*100, 3)}%")
    
    print(f"\nF1-Score (2 * (precicion * recall) / (precision + recall))")
    print(f"\"The harmonic mean/weighted average of precision and recall.\"")
    print(f"{round(metrics['f1_score']*100, 3)}%")

    print(f"\nAUC (Area under ROC curve, ROC-AUC)")
    print(f"\"Tells us about the capability of model in distinguishing the classes\"")
    print(f"{round(metrics['roc_auc']*100, 3)}%")