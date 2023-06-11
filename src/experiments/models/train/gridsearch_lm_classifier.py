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
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import ray
from ray import tune, cluster_resources
# from ray.air import session, RunConfig
# from ray.air.checkpoint import Checkpoint
# from ray.tune.experiment.trial import Trial
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.train.huggingface import HuggingFaceTrainer
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from csv import QUOTE_NONE
import csv
csv.field_size_limit(sys.maxsize)
import wandb
# wandb.login(key="57878dd06745f877fc0ce405c74e1a57103391f0") # TODO Make .env file for this key

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

    # Read csv
    df = pd.read_csv(datasets[dataset], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)

    # Filter out date and name
    df = df.drop(["date", "name"], axis=1)

    return df

def compute_metrics_for_classification(eval_pred):
    logits, labels = eval_pred
    predicted_labels = logits.argmax(axis=-1)

    # Get scores
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    f1score = 2 * (precision * recall) / (precision + recall)
    beta = 0.5
    f05score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    beta = 2
    f2score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    
    # Return dict of metrics
    return {
        "eval_tn": tn,
        "eval_fp": fp,
        "eval_fn": fn,
        "eval_tp": tp,
        "eval_accuracy": accuracy,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1score,
        "eval_f05": f05score,
        "eval_f2": f2score,
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
        # return 32

def train(
        config,
        X: List, y: List, 
        X_val: List = None, y_val: List = None, 
        val_portion: float = 0.2, 
        max_length: int = 512, 
        model_name: str = "distilbert-base-uncased", 
        saved_model_checkpoints: str = str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor" / "distilbert"), 
        log_path: str = "./logs",
        ):

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
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2, id2label=id2label, label2id=label2id).to(device)
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2, id2label=id2label, label2id=label2id).to(device)

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
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=64,
        weight_decay=config["weight_decay"],
        learning_rate=config["learning_rate"],
        save_total_limit = 10,
        load_best_model_at_end = True,
        metric_for_best_model = 'eval_f2',    
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        report_to='none',
        # report_to='wandb',  # enable reporting to W&B
    )

    trainer = Trainer(
        model_init = model_init,                         
        args = training_args,                  
        train_dataset = train_dataset,         
        eval_dataset = val_dataset,
        compute_metrics = compute_metrics_for_classification,
    )

    # Train the model
    trainer.train()
    
    # Evaluate model
    eval_metrics = trainer.evaluate()

    # Report eval metrics
    tune.report(**eval_metrics)
        

click.option = partial(click.option, show_default=True)
@click.command()
@click.option("-m", "--model", type=click.Choice(["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "albert-base-v2"]), default="distilbert-base-uncased", help="Model name")
@click.option("-s", "--size", type=click.Choice(["512", "256"]), default="512", help="Text max length")
@click.option("-d", "--dataset", type=click.Choice(datasets.keys()), default="train_sliced_stair_twitter_256", help="Dataset to use")
@click.option("--sample-size", type=float, default=None, help="Defined sample size of original size")
def main(model, size, dataset, sample_size):
    # Run parameters
    MAX_LENGTH = int(size)
    SAVED_MODEL_PATH = str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_classifier_grid" / model / dataset)
    LOG_PATH = "./logs"

    # Load data
    df_train = _get_dataframe(dataset=dataset)
    df_val = _get_dataframe(dataset=dataset.replace("train", "val"))
    # df_test = _get_dataframe(dataset=dataset.replace("train", "test"))
    
    # Set texts and labels
    X_train = df_train.text.values.tolist()
    y_train = df_train.label.values
    X_val = df_val.text.values.tolist()
    y_val = df_val.label.values
    # X_test = df_test.text.values.tolist()
    # y_test = df_test.label.values
    if sample_size:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=(1-sample_size), random_state=42, stratify=y_train)
        X_val, _, y_val, _ = train_test_split(X_val, y_val, test_size=(1-sample_size), random_state=42, stratify=y_val)
        # X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=(1-sample_size), random_state=42, stratify=y_test)

    ###########################
    ## Hyperparameter search ##
    ###########################
    ray.init(num_gpus=1)
    # ray.init()
    print(cluster_resources())

    # Set params
    num_samples=20
    max_num_epochs=10
    gpus_per_trial=1
    hyperparam_space = {
        "epochs": tune.choice([5, 7, 10]),
        "train_batch_size": tune.choice([32, 64]),
        "weight_decay": tune.choice([0.0, 0.01]),
        "learning_rate": tune.choice([1e-3, 1e-4, 2e-5, 3e-5, 5e-5]),
    }

    # Find number of available gpus
    def get_available_gpus():
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 0
    num_gpus = get_available_gpus()

    # Define scheduler
    scheduler = ASHAScheduler(
        metric="eval_f2",
        mode="max",
        max_t=max_num_epochs,
        grace_period=3,
        reduction_factor=3)
    
    # Define reporter
    reporter = CLIReporter(
        parameter_columns=list(hyperparam_space.keys()),
        metric_columns=["eval_f2", "eval_f1", "eval_recall", "eval_precision", "eval_tp", "eval_tn", "eval_fp", "eval_fn", "eval_loss", "epoch", "training_iteration"])

    try:
        result = tune.run(
            tune.with_parameters(train, X=X_train, y=y_train, X_val=X_val, y_val=y_val, max_length=MAX_LENGTH, model_name=model, saved_model_checkpoints=SAVED_MODEL_PATH, log_path=LOG_PATH),
            resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
            config=hyperparam_space,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir=str(Path(os.path.abspath(__file__)).parents[0] / "gs_lm_results" / "gs_4" / f"{model}_{size}"),
            name=f"{model}_{size}"
        )
    except ray.tune.error.TuneError as e:
        print("TuneError:", e)

    print("eval_f2")
    best_trial = result.get_best_trial("eval_f2", mode="max", scope="all")
    print(f"Best trial config: {best_trial.config}")
    print("Best metrics: ", best_trial.last_result)
    print(f"Best trial final validation loss: {best_trial.last_result['eval_loss']}")
    
    print("eval_loss")
    best_trial = result.get_best_trial("eval_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print("Best metrics: ", best_trial.last_result)
    print(f"Best trial final validation loss: {best_trial.last_result['eval_loss']}")

if __name__ == "__main__":
    main()