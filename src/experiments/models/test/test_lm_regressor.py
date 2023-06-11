# Imports
from functools import partial
import os
import sys
from pathlib import Path
from typing import List
import click
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from csv import QUOTE_NONE
import csv
csv.field_size_limit(sys.maxsize)
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, print_metrics_comprehensive, print_metrics_tabulated


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
    
    "shooter_hold_out": base_path / "shooter_hold_out.csv",
}
def _get_dataframe(dataset: str = "train_sliced_stair_twitter"):

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
    """Method for inference.

    Args:
        text (str): texts to run inference on.
        checkpoint (Path, optional): Checkpoint to load. Defaults to Path(os.path.abspath(__file__)).parents[1]/"saved_models"/"lm_regressor"/"bert_encodings"/"checkpoint-50".
        model_name (str, optional): Hugging Face model name. Defaults to "distilbert-base-uncased".
        max_length (int, optional): Maximum length for truncation. Defaults to 512.

    Returns:
        float: predicted score between 0 and 1
    """
    
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
        tokenizer,
        model, 
        thresholds: List[float] = [0.5],
        max_length: int = 512
        ):
    """Method for running test using a Hugging Face Transformer model.

    Args:
        texts (List[str]): texts for test.
        labels (List[str]): Matching labels for texts.
        tokenizer (AutoTokenizer): Tokenizer to use, downloaded from Hugging Face.
        model (AutoModelForSequenceClassification): Model to download from Hugging Face.
        thresholds (List[float], optional): If passed as input, run tests on thresholds. Defaults to None.
        max_length (int, optional): Maximum length for truncation. Defaults to 512.
    """

    # Forward pass for all texts
    model.eval()
    predictions = []
    
    for text in texts:
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        model_out = model(**encoding)
        pred = torch.sigmoid(model_out.logits).tolist()[0][0]
        predictions.append(pred)

    table = []
    keys = []
    for threshold in thresholds:
        keys.append(f"T({threshold})")
        table.append(get_metrics(predictions=[1 if pred >= threshold else 0 for pred in predictions], labels=labels))
    print_metrics_tabulated(keys=keys, list_of_metrics=table)

click.option = partial(click.option, show_default=True)
@click.command()
@click.option("-m", "--model", type=click.Choice(os.listdir(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor")), default="distilbert-base-uncased", help="Model from saved models")
@click.option("-c", "--checkpoint", default="checkpoint-2130", help="Checkpoint to use")
@click.option("-d", "--dataset", default="train_sliced_stair_twitter", help="Dataset used for fine-tuning")
@click.option("-t", "--test-file", default="test_sliced_stair_twitter", help="Test file")
@click.option("--max_len", default=512, help="Maximum length of texts")
def main(model, checkpoint, dataset, test_file, max_len):

    model_path = Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_regressor" / model
    assert os.path.isdir(model_path / f"{dataset}_{max_len}" / checkpoint), "Checkpoint not existing!"
    checkpoint = model_path / f"{dataset}_{max_len}" / checkpoint

    # "inference", "test"
    method = "test"

    # Data
    df = _get_dataframe(dataset=f"{test_file}_{max_len}")

    if (method == "inference"):
        
        # Run inference
        text = "Are you going to detect me? :)"
        print("Score:", inference(text, checkpoint=checkpoint, model_name=model, max_length=max_len))
    

    elif (method == "test"):

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model)
        saved_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 1, local_files_only=True)

        # Extract examples and labels (X and y)
        texts = list(df.text.values)
        labels = list(df.label.values)
        
        # Set thresholds to run for
        thresholds = np.round(np.arange(0.5, 0.601, 0.0025), 4)
        # thresholds = [0.5]

        # Run test
        test(texts, labels, tokenizer, saved_model, thresholds=thresholds, max_length = max_len)

        
if __name__ == "__main__":
    main()