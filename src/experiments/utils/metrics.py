from typing import List
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, recall_score, precision_score, f1_score
from collections import defaultdict
from tabulate import tabulate

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
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    specificity = (tn) / (tn + fp)
    f1score = 2 * (precision * recall) / (precision + recall)
    beta = 0.5
    f05score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    beta = 2
    f2score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    
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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1score,
        "f05_score": f05score,
        "f2_score": f2score,
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
    print(f"F0.5-score: {round(metrics['f05_score']*100, 3)}%")
    print(f"F2-score: {round(metrics['f2_score']*100, 3)}%")
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

    print(f"\nF0.5-Score (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)")
    print(f"\"Building on F1, this is the weighted average of precision and recall, with a bit more weight on precision. Beta equals 0.5.\"")
    print(f"{round(metrics['f05_score']*100, 3)}%")

    print(f"\nF2-Score (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)")
    print(f"\"Building on F1, this is the weighted average of precision and recall, with a bit more weight on recall. Beta equals 2.\"")
    print(f"{round(metrics['f2_score']*100, 3)}%")    

    print(f"\nAUC (Area under ROC curve, ROC-AUC)")
    print(f"\"Tells us about the capability of model in distinguishing the classes\"")
    print(f"{round(metrics['roc_auc']*100, 3)}%" if metrics['roc_auc'] is not None else "undefined")

def print_metrics_tabulated(keys: List, list_of_metrics: List[dict]):
    assert len(list_of_metrics) > 0, "List of metrics is empty!"
    assert len(keys) == len(list_of_metrics), "Keys need "

    headers = ["Key"] + list(list_of_metrics[0].keys())
    table = []
    for i, metrics in enumerate(list_of_metrics):
        table.append([keys[i]] + list(metrics.values()))

    print(tabulate(table, headers=headers))

def get_posts_ordered_by_confusion_matrix(texts: List[str], predictions: List[float], labels: List[float]):
    text_dicts = {"tp": [], "fp": [], "tn": [], "fn": []}
    for i, prediction in enumerate(predictions):
        if prediction == labels[i]: # True
            if labels[i] == 1: # True Positive
                text_dicts["tp"].append(texts[i])
            elif labels[i] == 0: # True Negative
                text_dicts["tn"].append(texts[i])
        else: # False
            if labels[i] == 0: # False Positive
                text_dicts["fp"].append(texts[i])
            elif labels[i] == 1: # False Negative
                text_dicts["fn"].append(texts[i])
    return text_dicts

def combined_recall_f1(y_true, y_pred, recall_weight=0.5):
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return recall_weight * recall + (1 - recall_weight) * f1

if __name__ == "__main__":

    ex = ["This is true negative", "This is false negative", "This is true positive", "This is false positive"]
    preds = [0, 0, 1, 1]
    labels = [0, 1, 1, 0]
    print(get_posts_ordered_by_confusion_matrix(ex, preds, labels))
    
    # metrics = {
    #     "tn": 1,
    #     "fp": 2,
    #     "fn": 3,
    #     "tp": 4,
    #     "accuracy": 5,
    #     "precision": 6,
    #     "recall": 7,
    #     "specificity": 8,
    #     "f1_score": 9,
    #     "roc_auc": None
    # }

    # metrics2 = {
    #     "tn": 1,
    #     "fp": 2,
    #     "fn": 3,
    #     "tp": 4,
    #     "accuracy": 5,
    #     "precision": 6,
    #     "recall": 7,
    #     "specificity": 8,
    #     "f1_score": 9,
    #     "roc_auc": 10
    # }

    # get_average_metrics([metrics, metrics2])
