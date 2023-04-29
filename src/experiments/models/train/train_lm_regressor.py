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
import pandas as pd
from csv import QUOTE_NONE
import csv
csv.field_size_limit(sys.maxsize)
import wandb
wandb.login(key="57878dd06745f877fc0ce405c74e1a57103391f0") # TODO Make .env file for this key

def _get_dataframe(dataset: str = "all_labeled"):
    # Extract basepath
    base_path = Path(os.path.abspath(__file__)).parents[3] / "dataset_creation" / "data"
    # Define all possible datasets
    datasets = {
        "train_sliced_stair_twitter": base_path / "train_test" / "train_sliced_stair_twitter.csv",
        "train_sliced_stair_twitter_256": base_path / "train_test" / "train_sliced_stair_twitter_256.csv",
        "train_no_stair_twitter": base_path / "train_test" / "train_no_stair_twitter.csv",
        "train_no_stair_twitter_256": base_path / "train_test" / "train_no_stair_twitter_256.csv",
        
        "test_sliced_stair_twitter": base_path / "train_test" / "test_sliced_stair_twitter.csv",
        "test_sliced_stair_twitter_256": base_path / "train_test" / "test_sliced_stair_twitter_256.csv",
        "test_no_stair_twitter": base_path / "train_test" / "test_no_stair_twitter.csv",
        "test_no_stair_twitter_256": base_path / "train_test" / "test_no_stair_twitter_256.csv",
        
        "shooter_hold_out_test": base_path / "train_test" / "shooter_hold_out_test.csv",
        "shooter_hold_out_test_256": base_path / "train_test" / "shooter_hold_out_test_256.csv",
    }

    # Read csv
    df = pd.read_csv(datasets[dataset], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)

    # Filter out date and name
    df = df.drop(["date", "name"], axis=1)

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

def train(
        X: List, y: List, 
        X_test: List = None, y_test: List = None, 
        val_portion: float = 0.2, 
        max_length: int = 512, 
        model_name: str = "distilbert-base-uncased", 
        num_epochs: int = 5, 
        saved_model_checkpoints: str = str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor" / "distilbert"), 
        log_path: str = "./logs"
        ):
    
    assert (X_test != None and y_test != None) or (X_test == None and y_test == None), "If test set defined, you need to pass in both!" 

    # Initialize device
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 1).to(device)

    # Split data into train and validation sets
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
        num_train_epochs = num_epochs,     
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

    # Trainer test metrics
    if X_test and y_test:
        test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_length)
        test_dataset = MakeTorchData(test_encodings, y_test.ravel())
        trainer.eval_dataset = test_dataset
        trainer.evaluate()

click.option = partial(click.option, show_default=True)
@click.command()
@click.option("-m", "--model", type=click.Choice(["distilbert-base-uncased", "bert-base-uncased"]), default="distilbert-base-uncased", help="Model name")
def main(model):
    # Parameters
    VAL_PORTION = 0.2
    MAX_LENGTH = 512
    NUM_EPOCHS = 5
    SAVED_MODEL_PATH = str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor" / model)
    LOG_PATH = "./logs"

    # Data
    df = _get_dataframe(dataset="train_sliced_stair_twitter")

    # Set X and y
    X = df.text.values.tolist()
    y = df.label.values

    # test_df = _get_dataframe(dataset="test")
    # X_test = test_df.text.values.tolist()
    # y_test = test_df.label.values

    train(X=X, y=y, val_portion=VAL_PORTION, max_length=MAX_LENGTH, model_name=model, num_epochs=NUM_EPOCHS, saved_model_checkpoints=SAVED_MODEL_PATH, log_path=LOG_PATH)

if __name__ == "__main__":
    main()