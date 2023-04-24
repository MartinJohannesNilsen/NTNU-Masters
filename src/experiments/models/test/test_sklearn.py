from functools import partial
import os
from pathlib import Path
import click
import numpy as np
import pickle
import sys
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, print_metrics_comprehensive
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


click.option = partial(click.option, show_default=True)
@click.command()
@click.argument("model_path", nargs=1)
@click.argument("test_path", nargs=1)
@click.option("-t", "--threshold", default=0.5, help="Threshold for predictions")
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
    main()