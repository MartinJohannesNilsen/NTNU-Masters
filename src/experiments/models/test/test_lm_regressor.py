# Imports
import os
import sys
from pathlib import Path
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from csv import QUOTE_NONE
import csv
csv.field_size_limit(sys.maxsize)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

def _get_dataframe(dataset: str = "all_labeled"):
    # Extract basepath
    base_path = Path(os.path.abspath(__file__)).parents[3] / "dataset_creation" / "data"
    # Define all possible datasets
    datasets = {
        "all_labeled": base_path / "all_labeled.csv",
        "all": base_path / "all.csv",
        "manifestos": base_path / "manifestos.csv",
        "mypersonality": base_path / "mypersonality.csv",
        "school_shooters": base_path / "school_shooters.csv",
        "stair_twitter_archive": base_path / "stair_twitter_archive.csv",
        "stream_of_consciousness": base_path / "stream_of_consciousness.csv",
        "twitter": base_path / "twitter.csv",
    }

    # Read csv
    all_labeled_df = pd.read_csv(datasets["all_labeled"], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)
    
    # Remove manifestos # TODO Might want to use these, but create smaller posts instead
    all_labeled_df = all_labeled_df[all_labeled_df.name != "manifestos"]

    # Filter out date and name
    df = all_labeled_df.drop(["date", "name"], axis=1)

    return df


def inference(
        text: str, 
        checkpoint: Path = Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor" / "bert_encodings" / "checkpoint-50",
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512
        ):
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 1, local_files_only=True)
    
    # Encode input
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

    # Forward pass
    model.eval()
    out = model(**encoding)
    pred = torch.sigmoid(out.logits).tolist()[0][0]
    return pred


def test(
        texts: List[str], 
        checkpoint=Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor" / "bert_encodings" / "checkpoint-50", 
        threshold=0.75,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512
        ):
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 1, local_files_only=True)

    # Forward pass for all texts
    model.eval()
    predictions = []
    y_pred_binary = []
    y_pred_threshold = []
    
    for text in texts:
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        out = model(**encoding)
        pred = torch.sigmoid(out.logits).tolist()[0][0]
        predictions.append(pred)
        y_pred_binary.append(1 if pred > 0.5 else 0)
        y_pred_threshold.append(1 if pred > threshold else 0)

    return pred, y_pred_binary, y_pred_threshold


if __name__ == "__main__":

    # "inference", "test"
    method = "test"

    # Data
    df = _get_dataframe()
    dev_df = df.sample(n=100)
    train_df = df.sample(frac=0.8)
    test_df = df.drop(train_df.index)
    df = dev_df
    
    if (method == "inference"):
        print(inference("Are you going to detect me? :)"))
    elif (method == "test"):
        texts = df.text.values
        labels = df.label.values
        threshold = 0.75
        outputs, y_pred_binary, y_pred_threshold = test(texts, 
                                                        threshold=threshold, 
                                                        model_name = "distilbert-base-uncased", 
                                                        max_length = 512)

        
        # Stats

        # Get binary stats
        tn, fp, fn, tp = confusion_matrix(labels, y_pred_binary).ravel()
        precision, recall, fscore, _ = precision_recall_fscore_support(labels, y_pred_binary, average="macro", zero_division=0)
        roc_auc = roc_auc_score(labels, y_pred_binary)
        binary_stats = {
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

        # Get threshold stats
        tn, fp, fn, tp = confusion_matrix(labels, y_pred_threshold).ravel()
        precision, recall, fscore, _ = precision_recall_fscore_support(labels, y_pred_threshold, average="macro", zero_division=0)
        roc_auc = roc_auc_score(labels, y_pred_threshold)
        threshold_stats = {
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

        # Print stats
        print("-"*5, f"Stats ", "-"*5)
        print(f"\nConfusion matrix")
        print(f"\"The number of True Positives, True Negatives, False Positives and False Negatives.\"")
        print(f"Binary:           TP: {binary_stats['tp']} | TN: {binary_stats['tn']} | FP: {binary_stats['fp']} | FN: {binary_stats['tp']}")
        print(f"Threshold = {threshold}: TP: {threshold_stats['tp']} | TN: {threshold_stats['tn']} | FP: {threshold_stats['fp']} | FN: {threshold_stats['fn']}")

        print(f"\nAccuracy ((TP + TN) / (TP + FP + FN + TN)")
        print(f"\"The percentage of all classifications which is true.\"")
        print(f"Binary: {round(binary_stats['accuracy']*100, 3)}%")
        print(f"Threshold = {threshold}: {round(threshold_stats['accuracy']*100, 3)}%")

        print(f"\nPrecision (TP / (TP + FP))")
        print(f"\"The percentage of classified positives which is true.\"")
        print(f"Binary: {round(binary_stats['precision']*100, 3)}%")
        print(f"Threshold = {threshold}: {round(threshold_stats['precision']*100, 3)}%")
        
        print(f"\nRecall (TP / (TP + FN))")
        print(f"\"Out of all the actual positives, how many did we correctly classify?\"")
        print(f"Binary: {round(binary_stats['recall']*100, 3)}%")
        print(f"Threshold = {threshold}: {round(threshold_stats['recall']*100, 3)}%")

        print(f"\nSpecificity (TN / (TN + FP))")
        print(f"\"Out of all the actual negatives, how many did we correctly classify?\"")
        print(f"Binary: {round(binary_stats['specificity']*100, 3)}%")
        print(f"Threshold = {threshold}: {round(threshold_stats['specificity']*100, 3)}%")
        
        print(f"\nF1-Score (2 * (precicion * recall) / (precision + recall))")
        print(f"\"The harmonic mean/weighted average of precision and recall.\"")
        print(f"Binary: {round(binary_stats['f1_score']*100, 3)}%")
        print(f"Threshold = {threshold}: {round(threshold_stats['f1_score']*100, 3)}%")

        print(f"\nAUC (Area under ROC curve, ROC-AUC)")
        print(f"\"Tells us about the capability of model in distinguishing the classes\"")
        print(f"Binary: {round(binary_stats['roc_auc']*100, 3)}%")
        print(f"Threshold = {threshold}: {round(threshold_stats['roc_auc']*100, 3)}%")