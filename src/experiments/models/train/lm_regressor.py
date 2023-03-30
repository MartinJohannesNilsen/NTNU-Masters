# Imports
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from csv import QUOTE_NONE
import csv
csv.field_size_limit(sys.maxsize)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

# Parameters
VAL_PORTION = 0.2
TEST_PORTION = 0.2
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
NUM_EPOCHS = 5
SAVED_MODEL_PATH = str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor" / "bert_encodings")
LOG_PATH = "./logs"


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

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    #smape = 1/len(labels) * np.sum(2 * np.abs(logits-labels) / (np.abs(labels) + np.abs(logits))*100)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
  
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "accuracy": accuracy} # "smape": smape

class MakeTorchData(torch.utils.data.Dataset):
    """Manipulate data to have label as float"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        item["labels"] = float(item["labels"])
        return item

    def __len__(self):
        return len(self.labels)

def train(df: pd.DataFrame = _get_dataframe(), test=False):

    # Initialize device
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = 1).to(device)

    # Set X and y
    X = df.text.values
    y = df.label.values

    # Split Data into Train, Val and Test
    if test: 
        X_train, X_test, y_train, y_test = train_test_split(X.tolist(), y, test_size=TEST_PORTION) # Train Test split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_PORTION) # Train Val split
    
    # Encode the text
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=MAX_LENGTH)

    # convert our tokenized data into a torch Dataset
    train_dataset = MakeTorchData(train_encodings, y_train.ravel())
    val_dataset = MakeTorchData(val_encodings, y_val.ravel())

    training_args = TrainingArguments(
        output_dir = SAVED_MODEL_PATH,          
        logging_dir = LOG_PATH,            
        num_train_epochs = NUM_EPOCHS,     
        per_device_train_batch_size = 32,   
        per_device_eval_batch_size = 20,   
        weight_decay = 0.01,               
        learning_rate = 2e-5,
        save_total_limit = 10,
        load_best_model_at_end = True,     
        metric_for_best_model = 'rmse',    
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
    )   

    trainer = Trainer(
        model = model,                         
        args = training_args,                  
        train_dataset = train_dataset,         
        eval_dataset = val_dataset,          
        compute_metrics = compute_metrics_for_regression,     
    )

    # Train the model
    trainer.train()

    # Call the summary
    trainer.evaluate()

    # Test on dedicated test dataset
    if test:
        test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LENGTH)
        test_dataset = MakeTorchData(test_encodings, y_test.ravel())
        trainer.eval_dataset = test_dataset
        trainer.evaluate()


def inference(text, checkpoint=Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor" / "bert_encodings" / "checkpoint-50"):
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 1, local_files_only=True)
    
    # Encode input
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)

    # Forward pass
    model.eval()
    out = model(**encoding)
    pred = torch.sigmoid(out.logits).tolist()[0][0]
    return pred


def test(texts, labels, checkpoint=Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor" / "bert_encodings" / "checkpoint-50", threshold=0.75):
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 1, local_files_only=True)

    # Forward pass for all texts
    model.eval()
    predictions = []
    y_pred_binary = []
    y_pred_threshold = []
    
    for i, (text, label) in enumerate(zip(texts, labels)):
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
        out = model(**encoding)
        pred = torch.sigmoid(out.logits).tolist()[0][0]
        predictions.append(pred)
        y_pred_binary.append(1 if pred > 0.5 else 0)
        y_pred_threshold.append(1 if pred > threshold else 0)

    return pred, y_pred_binary, y_pred_threshold


if __name__ == "__main__":

    # "train", "inference", "test"
    method = "test"

    # Data
    df = _get_dataframe()
    dev_df = df.sample(n=100)
    train_df = df.sample(frac=0.8)
    test_df = df.drop(train_df.index)
    df = dev_df
    
    if method == "train":
        train()
    elif (method == "inference"):
        print(inference("Are you going to detect me? :)"))
    elif (method == "test"):
        texts = df.text.values
        labels = df.label.values
        threshold = 0.75
        outputs, y_pred_binary, y_pred_threshold = test(texts, labels, threshold=threshold)
        
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
        print(f"\"The percentage of classifications which is in fact true.\"")
        print(f"Binary: {round(binary_stats['accuracy']*100, 3)}%")
        print(f"Threshold = {threshold}: {round(threshold_stats['accuracy']*100, 3)}%")

        print(f"\nPrecision (TP / (TP + FP))")
        print(f"\"The percentage of classified positives which is true.\"")
        print(f"Binary: {round(binary_stats['precision']*100, 3)}%")
        print(f"Threshold = {threshold}: {round(threshold_stats['precision']*100, 3)}%")
        
        print(f"\nRecall (TP / (TP + FN))")
        print(f"\"Out of all the positives, how many did we correctly classify?\"")
        print(f"Binary: {round(binary_stats['recall']*100, 3)}%")
        print(f"Threshold = {threshold}: {round(threshold_stats['recall']*100, 3)}%")

        print(f"\nSpecificity (TN / (TN + FP))")
        print(f"\"Out of all the negatives, how many did we correctly classify?\"")
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