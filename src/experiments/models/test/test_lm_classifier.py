# Imports
from functools import partial
import os
import sys
from pathlib import Path
from typing import List
import click
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from csv import QUOTE_NONE
import csv
csv.field_size_limit(sys.maxsize)
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, print_metrics_comprehensive, print_metrics_tabulated


base_path = Path(os.path.abspath(__file__)).parents[3] / "dataset_creation" / "data" / "train_test"
datasets = {
    "train_sliced_stair_twitter_512": base_path / "new" / "train_sliced_stair_twitter_512.csv",
    "train_sliced_stair_twitter_256": base_path / "new" / "train_sliced_stair_twitter_256.csv",
    "train_no_stair_twitter_512": base_path / "new" / "train_no_stair_twitter_512.csv",
    "train_no_stair_twitter_256": base_path / "new" / "train_no_stair_twitter_256.csv",
    
    "test_sliced_stair_twitter_512": base_path / "new" / "test_sliced_stair_twitter_512.csv",
    "test_sliced_stair_twitter_256": base_path / "new" / "test_sliced_stair_twitter_256.csv",
    "test_no_stair_twitter_512": base_path / "new" / "test_no_stair_twitter_512.csv",
    "test_no_stair_twitter_256": base_path / "new" / "test_no_stair_twitter_256.csv",
    
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
        classifier,
        output_scores = False,
        max_length: int = 512
        ):

    # Forward pass
    result = classifier(text)
    score = result[0]['score'][1]
    pred = result[0]['label']
    if output_scores:
        return score
    else:
        return pred


def test(
        texts: List[str],
        labels: List[str], 
        classifier,
        output_scores = False,
        thresholds: List[float] = [0.5],
        max_length: int = 512
        ):

    # Forward pass for all texts
    predictions = []
    
    for text in texts:
        result = classifier(text)
        score = result[0]['score'][1]
        pred = result[0]['label']
        if output_scores:
            predictions.append(score)
        else:
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
        saved_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2, local_files_only=True)
        # model = torch.load(checkpoint)
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

        # Extract examples and labels (X and y)
        texts = list(df.text.values)
        labels = list(df.label.values)
        
        # Set thresholds to run for
        thresholds = np.round(np.arange(0.5, 0.601, 0.0025), 4)
        # thresholds = [0.5]

        # Run test
        test(texts, labels, classifier, thresholds=thresholds, max_length = max_len)

        
if __name__ == "__main__":
    main()