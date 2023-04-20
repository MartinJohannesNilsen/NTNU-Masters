# Imports
from functools import partial
import os
import sys
from pathlib import Path
from typing import List
import click
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from csv import QUOTE_NONE
import csv
csv.field_size_limit(sys.maxsize)
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, print_metrics_comprehensive


def _get_dataframe(dataset: str = "all_labeled"):
    # Extract basepath
    base_path = Path(os.path.abspath(__file__)).parents[3] / "dataset_creation" / "data"
    # Define all possible datasets
    datasets = {
        "train_sliced_stair_twitter": base_path / "train_test" / "train_sliced_stair_twitter.csv",
        "test_sliced_stair_twitter": base_path / "train_test" / "test_sliced_stair_twitter.csv",
        "hold_out_test_sliced_stair_twitter": base_path / "train_test" / "hold_out_test_sliced_stair_twitter.csv",
        # "train_no_stair_twitter": base_path / "train_test" / "train_no_stair_twitter.csv",
        # "test_no_stair_twitter": base_path / "train_test" / "test_no_stair_twitter.csv",
        # "all_labeled": base_path / "all_labeled.csv",
        # "all": base_path / "all.csv",
        # "manifestos": base_path / "manifestos.csv",
        # "mypersonality": base_path / "mypersonality.csv",
        # "school_shooters": base_path / "school_shooters.csv",
        # "stair_twitter_archive": base_path / "stair_twitter_archive.csv",
        # "stream_of_consciousness": base_path / "stream_of_consciousness.csv",
        # "twitter": base_path / "twitter.csv",
    }

    # Read csv
    df = pd.read_csv(datasets[dataset], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)

    # Filter out date and name
    df = df.drop(["date", "name"], axis=1)

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
        labels: List[str], 
        checkpoint: Path = Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor" / "distilbert-base-uncased" / "checkpoint-50", 
        thresholds: List[float] = [0.5],
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512
        ):
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 1, local_files_only=True)

    # Forward pass for all texts
    model.eval()
    predictions = []
    
    for text in texts:
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        model_out = model(**encoding)
        pred = torch.sigmoid(model_out.logits).tolist()[0][0]
        predictions.append(pred)

    for threshold in thresholds:
        print(f"\nThreshold = {threshold}")
        print_metrics_comprehensive(get_metrics(predictions=[1 if pred > threshold else 0 for pred in predictions], labels=labels))

click.option = partial(click.option, show_default=True)
@click.command()
@click.option("-m", "--model", type=click.Choice(os.listdir(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor")), default="distilbert-base-uncased", help="Model from saved models")
@click.option("-c", "--checkpoint", default="checkpoint-2130", help="Checkpoint to use")
@click.option("--max_len", default=512, help="maximum length of texts")
def main(model, checkpoint, max_len):

    model_path = Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor" / model
    assert os.path.isdir(model_path / checkpoint), "Checkpoint not existing!"
    checkpoint = model_path / checkpoint

    # "inference", "test"
    method = "test"

    # Data
    df = _get_dataframe(dataset="test_sliced_stair_twitter")

    if (method == "inference"):
        print(inference("Are you going to detect me? :)", checkpoint=checkpoint))
    elif (method == "test"):
        texts = df.text.values
        labels = df.label.values
        thresholds = [0.7, 0.71, 0.72, 0.73, 0.74, 0.75]
        test(texts, labels, checkpoint=checkpoint, thresholds=thresholds, model_name = model, max_length = max_len)

        
if __name__ == "__main__":
    main()