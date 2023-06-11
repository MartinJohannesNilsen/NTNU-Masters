# Imports
from functools import partial
import os
import sys
from pathlib import Path
from typing import List
import click
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoConfig
import pandas as pd
from csv import QUOTE_NONE
import csv
csv.field_size_limit(sys.maxsize)
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, print_metrics_comprehensive, print_metrics_tabulated, print_metrics_simplified


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
    
    "shooter_hold_out_256": base_path / "shooter_hold_out_256.csv",
    "shooter_hold_out_512": base_path / "shooter_hold_out_512.csv",
}
def _get_dataframe(dataset: str = "train_sliced_stair_twitter"):
    """Helper function to get dataframe"""

    # Read csv
    df = pd.read_csv(datasets[dataset], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)

    # Filter out date and name
    df = df.drop(["date", "name"], axis=1)

    return df


def inference(
        text: str, 
        classifier,
        output_scores = False,
        tokenizer_kwargs = None
        ):
    """Method for running inference.

    Args:
        text (str): _description_
        classifier (): _description_
        output_scores (bool, optional): _description_. Defaults to False.
        tokenizer_kwargs (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    classification_output = classifier(text, **tokenizer_kwargs)

    # Extract label
    label = classification_output[0]['label']
    pred = 1 if label == "POSITIVE" else 0
    # print(label)

    # Extract score
    # As this is the probability of being that label, we need to translate negative score to (1-negative) for 0.99 negative to be 0.01 positive. 
    score = classification_output[0]['score']
    if label == "NEGATIVE":
        score = 1 - score
    # print(score)

    if output_scores:
        return score
    else:
        return pred


def test(
        texts: List[str],
        labels: List[str], 
        classifier,
        output_scores = False,
        thresholds: List[float] = None,
        tokenizer_kwargs = None
        ):
    """Method for running test using a Hugging Face Transformer model.

    Args:
        texts (List[str]): texts for test.
        labels (List[str]): Matching labels for texts.
        classifier (Pipeline): Loaded classifier pipeline.
        output_scores (bool, optional): If True, output scores instead of class labels. Defaults to False.
        thresholds (List[float], optional): If passed as input, run tests on thresholds. Defaults to None.
        tokenizer_kwargs (_type_, optional): Optional tokenizer arguments. Defaults to None.

    Returns:
        (List[int], List[float]): Tuple in the format of predicted labels and predicted scores.
    """

    # Forward pass for all texts
    pred_labels = []
    pred_scores = []
    
    for text in texts:
        classification_output = classifier(text, **tokenizer_kwargs)

        # Extract label
        label = classification_output[0]['label']
        pred = 1 if label == "POSITIVE" else 0
        # print(label)

        # Extract score
        # As this is the probability of being that label, we need to translate negative score to (1-negative) for 0.99 negative to be 0.01 positive. 
        score = classification_output[0]['score']
        if label == "NEGATIVE":
            score = 1 - score
        # print(score)

        pred_labels.append(pred)
        pred_scores.append(score)

    table = []
    keys = []
    if output_scores:
        if thresholds is not None:
            for threshold in thresholds:
                keys.append(f"T({threshold})")
                table.append(get_metrics(predictions=[1 if pred >= threshold else 0 for pred in pred_scores], labels=labels))
            print_metrics_tabulated(keys=keys, list_of_metrics=table)
        else:
            print_metrics_simplified(get_metrics(predictions=[1 if pred >= 0.5 else 0 for pred in pred_scores], labels=labels))
    else: 
        print_metrics_simplified(get_metrics(predictions=pred_labels, labels=labels))
    
    return pred_labels, pred_scores
    

click.option = partial(click.option, show_default=True)
@click.command()
@click.option("-m", "--model", type=click.Choice(os.listdir(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_classifier")), default="distilbert-base-uncased", help="Model from saved models")
@click.option("-c", "--checkpoint", default="final", help="Checkpoint to use")
@click.option("-d", "--dataset", default="train_sliced_stair_twitter_512", help="Dataset used for fine-tuning")
@click.option("-t", "--test-file", default="test_sliced_stair_twitter_512", help="Test file")
@click.option("--max_len", default=512, help="Maximum length of texts")
def main(model, checkpoint, dataset, test_file, max_len):

    model_name = model
    model_path = Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "lm_classifier" / model_name
    print(model_path / dataset / checkpoint)
    assert os.path.isdir(model_path / dataset / checkpoint), "Checkpoint not existing!"
    checkpoint = model_path / dataset / checkpoint

    # Data
    df = _get_dataframe(dataset=f"{test_file}")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2, local_files_only=True)
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':max_len}
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Extract examples and labels (X and y)
    texts = list(df.text.values)
    labels = list(df.label.values)
    
    # Set thresholds to run for
    # thresholds = np.round(np.arange(0.5, 0.601, 0.0025), 4)
    thresholds = np.round(np.arange(0, 1.01, 0.05), 2)
    # thresholds = [0.5]

    # Run test
    # test(texts, labels, classifier, output_scores=True, thresholds=thresholds, tokenizer_kwargs=tokenizer_kwargs)
    pred_labels, pred_scores = test(texts, labels, classifier, output_scores=True, tokenizer_kwargs=tokenizer_kwargs)

    out_path = str(Path(os.path.abspath(__file__)).parent / "LM_files" / f"{model_name}_{max_len}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("idx,pred_val,pred_label,label\n")
        
        for i, (score, pred, label) in enumerate(zip(pred_scores, pred_labels, labels)):
            f.write(f"{i},{score},{pred},{label}\n")


        
if __name__ == "__main__":
    main()