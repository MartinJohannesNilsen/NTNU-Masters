from typing import List
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from collections import defaultdict


def get_metrics(predictions: List[float], labels: List[float]) -> dict:
    """Get a selection of performance metrics.

    Args:
        predictions (List[float]): List of predictions.
        labels (List[float]): List of labels.

    Returns:
        dict: Dictionary of performance metrics.
    """
    
    # Gather performance metrics
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    try:
        roc_auc = roc_auc_score(labels, predictions)
    except:
        roc_auc = None
    
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

def get_average_metrics(metrics_array: List[dict]):

    # Create a combined dictionary with all values
    combined_dict = defaultdict(list)
    for d in metrics_array:
        for key, value in d.items():
            if value is not None:
                combined_dict[key].append(value)

    # Create average dictionary
    average = {}
    for key, value in combined_dict.items():
        average[key] = sum(value) / len(value)

    return average


def print_metrics_simplified(metrics: dict):
    print(f"TP: {metrics['tp']} | TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
    print(f"Accuracy: {round(metrics['accuracy']*100, 3)}%")
    print(f"Precision: {round(metrics['precision']*100, 3)}%")
    print(f"Recall: {round(metrics['recall']*100, 3)}%")
    print(f"Specificity: {round(metrics['specificity']*100, 3)}%")
    print(f"F1-score: {round(metrics['f1_score']*100, 3)}%")
    print(f"AUC: {round(metrics['roc_auc']*100, 3)}%" if metrics['roc_auc'] is not None else "AUC: undefined")


def print_metrics_comprehensive(metrics: dict):

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
    print(f"{round(metrics['roc_auc']*100, 3)}%" if metrics['roc_auc'] is not None else "undefined")

if __name__ == "__main__":
    
    metrics = {
        "tn": 1,
        "fp": 2,
        "fn": 3,
        "tp": 4,
        "accuracy": 5,
        "precision": 6,
        "recall": 7,
        "specificity": 8,
        "f1_score": 9,
        "roc_auc": None
    }

    metrics2 = {
        "tn": 1,
        "fp": 2,
        "fn": 3,
        "tp": 4,
        "accuracy": 5,
        "precision": 6,
        "recall": 7,
        "specificity": 8,
        "f1_score": 9,
        "roc_auc": 10
    }

    get_average_metrics([metrics, metrics2])
