# Imports
from functools import partial
import os
import sys
from pathlib import Path
from typing import List
import click
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from csv import QUOTE_NONE
import csv
csv.field_size_limit(sys.maxsize)

base_path = Path(os.path.abspath(__file__)).parents[4] / "data" / "processed_data" / "train_test" / "csv"
datasets = {
    "train_sliced_stair_twitter_512": base_path / "train_sliced_stair_twitter_512.csv",
    "train_sliced_stair_twitter_256": base_path / "train_sliced_stair_twitter_256.csv",
    "train_no_stair_twitter_512": base_path / "train_no_stair_twitter_512.csv",
    "train_no_stair_twitter_256": base_path / "train_no_stair_twitter_256.csv",
    
    "test_sliced_stair_twitter_512": base_path / "test_sliced_stair_twitter_512.csv",
    "test_sliced_stair_twitter_256": base_path / "test_sliced_stair_twitter_256.csv",
    "test_no_stair_twitter_512": base_path / "test_no_stair_twitter_512.csv",
    "test_no_stair_twitter_256": base_path / "test_no_stair_twitter_256.csv",
    
    "val_sliced_stair_twitter_512": base_path / "val_sliced_stair_twitter_512.csv",
    "val_sliced_stair_twitter_256": base_path / "val_sliced_stair_twitter_256.csv",
    "val_no_stair_twitter_512": base_path / "val_no_stair_twitter_512.csv",
    "val_no_stair_twitter_256": base_path / "val_no_stair_twitter_256.csv",

    "shooter_hold_out": base_path / "shooter_hold_out.csv",
    
}

def _get_dataframe(dataset: str = "train_sliced_stair_twitter"):
    """Helper function for getting dataframe"""

    # Read csv
    df = pd.read_csv(datasets[dataset], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)

    # Filter out date and name
    df = df.drop(["date", "name"], axis=1)

    return df

def compute_metrics_for_classification(eval_pred):
    """Helper function for metrics used during training"""
    logits, labels = eval_pred
    predicted_labels = logits.argmax(axis=-1)

    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels)
    recall = recall_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)
    f0_5 = fbeta_score(labels, predicted_labels, beta=0.5)
    f2 = fbeta_score(labels, predicted_labels, beta=2)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f0.5": f0_5,
        "f2": f2
    }

class MakeTorchData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train(
        X: List, y: List,
        X_val: List = None, y_val: List = None, 
        X_test: List = None, y_test: List = None, 
        val_portion: float = 0.2, 
        max_length: int = 512, 
        model_name: str = "distilbert-base-uncased", 
        saved_model_checkpoints: str = str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_classifier" / "distilbert-base-uncased"), 
        log_path: str = "./logs",
        configs = {"epochs": 5, "train_batch_size": 32, "weight_decay": 0.01, "learning_rate": 1e-5}
        ):
    """Method for training Hugging Face Transformer models with classification head.

    Args:
        X (List): Training samples.
        y (List): Training labels.
        X_val (List, optional): Validation samples. Defaults to None.
        y_val (List, optional): Validation labels. Defaults to None.
        X_test (List, optional): Testing samples. Defaults to None.
        y_test (List, optional): Testing labels. Defaults to None.
        val_portion (float, optional): Portion to use for validation. Defaults to 0.2.
        max_length (int, optional): Maximum length to use for truncation. Defaults to 512.
        model_name (str, optional): Model to download from Hugging Face. Defaults to "distilbert-base-uncased".
        saved_model_checkpoints (str, optional): Stringified path used for model checkpoint saving. Defaults to str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_classifier" / "distilbert-base-uncased").
        log_path (str, optional): Path to write logs. Defaults to "./logs".
        configs (dict, optional): Dictionary of model configurations. Defaults to {"epochs": 5, "train_batch_size": 32, "weight_decay": 0.01, "learning_rate": 1e-5}.
    """

    # Initialize device
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)
    
    # Initialize tokenizer and model
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2, id2label=id2label, label2id=label2id).to(device)

    # Split data into train and validation sets
    if X_val is not None and y_val is not None:
        X_train, y_train = X, y
    else:
        if X_val is not None or y_val is not None:
            print("Both X_val and y_val need to be input, defaulting to val_portion split of train!")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_portion) # Train Val split
    
    # Encode the text
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=max_length)

    # convert our tokenized data into a torch Dataset
    train_dataset = MakeTorchData(train_encodings, y_train.ravel())
    val_dataset = MakeTorchData(val_encodings, y_val.ravel())

    training_args = TrainingArguments(
        output_dir = saved_model_checkpoints,          
        logging_dir = log_path,            
        per_device_eval_batch_size = 64,
        per_device_train_batch_size = configs["train_batch_size"],
        num_train_epochs = configs["epochs"],     
        weight_decay = configs["weight_decay"],
        learning_rate = configs["learning_rate"],
        save_total_limit = 10,
        load_best_model_at_end = True,     
        metric_for_best_model = 'f2',    
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
    )   

    trainer = Trainer(
        model = model,                         
        args = training_args,                  
        train_dataset = train_dataset,         
        eval_dataset = val_dataset,
        compute_metrics = compute_metrics_for_classification,
    )

    # Train the model
    trainer.train()

    # Call the summary
    trainer.evaluate()

    # Trainer test metrics
    if X_test is not None and y_test is not None:
        print("-"*30)
        print("Evaluation on test dataset: ")
        print("-"*30)
        test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_length)
        test_dataset = MakeTorchData(test_encodings, y_test.ravel())
        trainer.eval_dataset = test_dataset
        trainer.evaluate()
        

click.option = partial(click.option, show_default=True)
@click.command()
@click.option("-m", "--model", type=click.Choice(["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "albert-base-v2"]), default="distilbert-base-uncased", help="Model name")
@click.option("-s", "--size", type=click.Choice(["512", "256"]), default="512", help="Text max length")
@click.option("-d", "--dataset", type=click.Choice(datasets.keys()), default="train_sliced_stair_twitter_256", help="Dataset to use")
def main(model, size, dataset):
    # Parameters
    MAX_LENGTH = int(size)
    SAVED_MODEL_PATH = str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_classifier" / model / dataset)
    LOG_PATH = "./logs"

    # Data
    df_train = _get_dataframe(dataset=dataset)
    df_val = _get_dataframe(dataset=dataset.replace("train", "val"))
    df_test = _get_dataframe(dataset=dataset.replace("train", "test"))

    # Set X and y
    X_train = df_train.text.values.tolist()
    y_train = df_train.label.values
    X_val = df_val.text.values.tolist()
    y_val = df_val.label.values
    X_test = df_test.text.values.tolist()
    y_test = df_test.label.values

    # Define hyperparams
    hyperparams = {"epochs": 5, "train_batch_size": 32, "weight_decay": 0.01, "learning_rate": 1e-5}

    # From gridsearch
    hyperparams_from_gridsearch = {
        "albert-base-v2_256": {'epochs': 7, 'train_batch_size': 32, 'weight_decay': 0.0, 'learning_rate': 3e-05},
        "albert-base-v2_512": {"epochs": 5, "train_batch_size": 32, "weight_decay": 0.01, "learning_rate": 2e-05},
        "bert-base-uncased_256": {'epochs': 7, 'train_batch_size': 32, 'weight_decay': 0.01, 'learning_rate': 3e-05},
        "bert-base-uncased_512": {"epochs": 7, "train_batch_size": 32, "weight_decay": 0.01, "learning_rate": 2e-05},
        "distilbert-base-uncased_256": {'epochs': 7, 'train_batch_size': 32, 'weight_decay': 0.01, 'learning_rate': 3e-05},
        "distilbert-base-uncased_512": {"epochs": 7, "train_batch_size": 32, "weight_decay": 0.0, "learning_rate": 3e-05},
        "roberta-base_256": {'epochs': 7, 'train_batch_size': 64, 'weight_decay': 0.0, 'learning_rate': 2e-05},
        "roberta-base_512": {"epochs": 10, "train_batch_size": 32, "weight_decay": 0.0, "learning_rate": 2e-05},
    }
    hyperparams = hyperparams_from_gridsearch[f"{model}_{size}"]
    print("Hyperparams:", hyperparams)
    train(X=X_train, y=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, max_length=MAX_LENGTH, model_name=model, saved_model_checkpoints=SAVED_MODEL_PATH, log_path=LOG_PATH, configs=hyperparams)

if __name__ == "__main__":
    main()