from csv import QUOTE_NONE
from functools import partial
import os
from pathlib import Path
import click
import numpy as np
import pickle
import sys
import pandas as pd
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, print_metrics_comprehensive, get_posts_ordered_by_confusion_matrix
from experiments.features.load_and_store_emb_batches import read_h5


def test_embeddings(model_path, test_path, batch_size: int = None):

    # Load the model from disk
    model = pickle.load(open(model_path, 'rb'))

    if not batch_size:
        data = read_h5(test_path)
        X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
        y_true = np.array(data["label"])
        y_pred = model.predict(X)
    else:
        n_total_samples = len(read_h5(test_path, col_name="idx"))
        y_pred = []
        y_true = []
        for i in range(0, n_total_samples, batch_size):
            data = read_h5(test_path, start=i, chunk_size=batch_size)
            X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
            y_pred.extend(model.predict(X))
            y_true.extend(data["label"])

    return y_pred, y_true


def test_liwc(model_path, test_path, batch_size: int = None):

    # Load the model from disk
    model = pickle.load(open(model_path, 'rb'))

    if not batch_size:
        data = read_h5(test_path)
        X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
        y_true = np.array(data["label"])
        y_pred = model.predict(X)
    else:
        n_total_samples = len(read_h5(test_path, col_name="idx"))
        y_pred = []
        y_true = []
        for i in range(0, n_total_samples, batch_size):
            data = read_h5(test_path, start=i, chunk_size=batch_size)
            X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
            y_pred.extend(model.predict(X))
            y_true.extend(data["label"])

    return y_pred, y_true

def get_texts_matching_tensors(emb_features_path, matching_dataset_path) -> pd.DataFrame:
    indexes = read_h5(emb_features_path, col_name="idx")
    dataset = pd.read_csv(matching_dataset_path, sep="‎", quoting=QUOTE_NONE, engine="python")
    filtered_df = dataset.loc[dataset.index[indexes]]
    assert len(filtered_df.index) == len(indexes), "Could not align h5 indices with original dataset!"
    return filtered_df["text"].to_list()


def get_texts_matching_liwc(liwc_features_path, matching_dataset_path) -> pd.DataFrame:
    # As liwc features do not filter out, we can return matching dataset path. Just ensure same length
    features = pd.read(liwc_features_path)
    dataset = pd.read_csv(matching_dataset_path, sep="‎", quoting=QUOTE_NONE, engine="python")
    assert len(features.index) == len(features.index), "Could not align liwc indices with original dataset!"
    return dataset["text"].to_list()


click.option = partial(click.option, show_default=True)
@click.command()
@click.argument("model_path", nargs=1)
@click.argument("test_path", nargs=1)
@click.option("-t", "--threshold", type=int, default=0.5, help="Threshold for predictions")
@click.option("-o", "--output", type=str, help="Output posts grouped by confusion matrix to file")
def main(model_path, test_path, threshold):
    assert threshold >= 0 and threshold <= 1, "Threshold needs to be between 0 and 1!"

    # Check that paths leads to files
    assert os.path.isfile(model_path), "No model file found!"
    assert os.path.isfile(test_path), "No test file found!"

    if "embeddings" in model_path:
        assert "embeddings" in test_path, "Mismatch between model and test embeddings!"
        y_pred, y_true = test_embeddings(model_path, test_path)
        out = [1 if pred > threshold else 0 for pred in y_pred]
        stats = get_metrics(out, y_true)
        print_metrics_comprehensive(stats)
    
    elif "liwc" in model_path:
        assert "liwc" in test_path, "Mismatch between model and test LIWC dictionaries!"
        y_pred, y_true = test_liwc(model_path, test_path)
        out = [1 if pred > threshold else 0 for pred in y_pred]
        stats = get_metrics(out, y_true)
        print_metrics_comprehensive(stats)


if __name__ == "__main__":
    # main()

    emb = "/cluster/home/martijni/NTNU-Masters/src/experiments/features/embeddings/train_sliced_stair_twitter_glove_50_head.h5"
    dataset = "/cluster/home/martijni/NTNU-Masters/src/dataset_creation/data/train_test/train_sliced_stair_twitter.csv"
    print(get_texts_matching_tensors(emb, dataset).head())