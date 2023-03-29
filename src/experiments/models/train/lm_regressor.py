# Imports
import os
from pathlib import Path
import sys
import torch
from transformers.file_utils import is_tf_available, is_torch_available
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
#%load_ext memory_profiler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import csv
csv.field_size_limit(sys.maxsize)
from csv import QUOTE_NONE

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
    outputs = []
    binary_metrics = {
        "tp": [],
        "tn": [],
        "fp": [],
        "fn": [],
    }
    threshold_metrics = {
        "tp": [],
        "tn": [],
        "fp": [],
        "fn": [],
    }
    for i, (text, label) in enumerate(zip(texts, labels)):
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
        out = model(**encoding)
        pred = torch.sigmoid(out.logits).tolist()[0][0]
        outputs.append(pred)

        # Return metrics based on both binary outcome and threshold
        # Binary
        if round(pred) == label:
            binary_metrics["tp"].append((i, pred)) if label == 1 else binary_metrics["tn"].append((i, pred))
        else:
            binary_metrics["fp"].append((i, pred)) if round(pred) == 1 else binary_metrics["fn"].append((i, pred))
        
        # Threshold
        if pred > threshold:
            threshold_metrics["tp"].append((i, pred)) if label == 1 else threshold_metrics["fp"].append((i, pred))
        else:
            threshold_metrics["fn"].append((i, pred)) if label == 1 else threshold_metrics["tn"].append((i, pred))
        
    
    return outputs, binary_metrics, threshold_metrics


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
        outputs, binary_metrics, threshold_metrics = test(texts, labels, threshold=threshold)
        
        # Stats
        print("-"*5, f"Stats ", "-"*5)
        # Precision
        print(f"\nPrecision")
        print(f"Binary")
        print(f"TP: {len(binary_metrics['tp'])}, TN: {len(binary_metrics['tn'])}, FP: {len(binary_metrics['fp'])}, FN: {len(binary_metrics['fn'])}")
        print(f"Threshold = {threshold}")
        print(f"TP: {len(threshold_metrics['tp'])}, TN: {len(threshold_metrics['tn'])}, FP: {len(threshold_metrics['fp'])}, FN: {len(threshold_metrics['fn'])}")
        
        # Recall
        print(f"\nRecall")
        n_threats = list(labels).count(1)
        print(f"Percentage of threatening texts identified (binary): {len(binary_metrics['tp'])}/{n_threats} ({round((len(binary_metrics['tp'])/n_threats if n_threats > 0 else 1)*100, 2)}%)")
        print(f"Percentage of threatening texts identified (threshold={threshold}): {len(threshold_metrics['tp'])}/{n_threats} ({round((len(threshold_metrics['tp'])/n_threats if n_threats > 0 else 1)*100, 2)}%)")
