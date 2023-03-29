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

def train(df: pd.DataFrame = _get_dataframe().head(500)):

    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)

    X = df.text.values
    y = df.label.values

    # Split Data into Train, Val and Test
    # Train Test
    X_train, X_test, y_train, y_test = train_test_split(X.tolist(), y, test_size=TEST_PORTION)
    # Train Val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_PORTION)

    # Call the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Encode the text
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LENGTH)

    # convert our tokenized data into a torch Dataset
    train_dataset = MakeTorchData(train_encodings, y_train.ravel())
    val_dataset = MakeTorchData(val_encodings, y_val.ravel())
    test_dataset = MakeTorchData(test_encodings, y_test.ravel())

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = 1).to(device)

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

    # Test on testy dataset
    trainer.eval_dataset = test_dataset
    trainer.evaluate()

def inference(text, checkpoint="results/checkpoint-50"):
    
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


def test(texts, labels, checkpoint="results/checkpoint-50", threshold=0.75):
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 1, local_files_only=True)

    # Forward pass for all texts
    model.eval()
    outputs = []
    correct_binary = []
    flagged_as_threat = []
    correctly_flagged = []
    incorrectly_flagged = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
        out = model(**encoding)
        pred = torch.sigmoid(out.logits).tolist()[0][0]
        outputs.append(pred)

        # Additional test outputs
        if round(pred) == label:
            correct_binary.append(i)
        
        if pred > threshold:
            flagged_as_threat.append(i)
            correctly_flagged.append(i) if label == 1 else incorrectly_flagged.append(i)
    
    return outputs, correct_binary, flagged_as_threat, correctly_flagged, incorrectly_flagged


if __name__ == "__main__":

    # Data
    df = _get_dataframe().head(10)
    
    # Train
    # train()

    # Inference
    # print(inference("Hello there, this is going marvelous!"))
    
    # Test
    texts = df.text.values
    labels = df.label.values
    outputs, correct_binary, flagged_as_threat, correctly_flagged, incorrectly_flagged = test(texts, labels)
    
    # Stats
    # Precision
    print(f"Precision")
    print(f"Number of correctly binary classifications: {len(correct_binary)}/{len(outputs)} ({(len(correct_binary)/len(outputs)*100)}%)")
    print(f"Number of flags: {len(flagged_as_threat)}")
    
    # print(f"True positives: {correctly_flagged)}/{len(flagged_as_threat)} ({(len(correctly_flagged)/len(outputs)*100)}%)")
    # print(f"True negatives: {len(incorrectly_flagged)}/{len(flagged_as_threat)} ({(len(incorrectly_flagged)/len(outputs)*100)}%)")
    # print(f"False positives: {len(incorrectly_flagged)}/{len(flagged_as_threat)} ({(len(incorrectly_flagged)/len(outputs)*100)}%)")
    # print(f"False negatives: {len(incorrectly_flagged)}/{len(flagged_as_threat)} ({(len(incorrectly_flagged)/len(outputs)*100)}%)")
    
    # Recall
    print(f"Recall")
    n_threats = list(labels).count(1)
    print(f"Percentage of threats identified: {len(correctly_flagged)}/{n_threats} ({(len(correctly_flagged)/n_threats if n_threats > 0 else 1)*100}%)")