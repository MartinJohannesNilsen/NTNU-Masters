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

EMB_TYPES = ["glove", "fasttext", "bert"]
def _get_model_path(model_name: str, embedding_type: str):
    assert embedding_type in EMB_TYPES, "Embedding not supported!"

    MODEL_PATHS = {
        "xgboost": Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'xgboost' / f'{embedding_type}_embeddings',
        "svm": Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'svm' / f'{embedding_type}_embeddings',
        "naive_bayes": Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'naive_bayes' / f'{embedding_type}_embeddings',
        "knn": Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'naive_bayes' / f'{embedding_type}_embeddings',
    }
    assert model_name in MODEL_PATHS.keys(), "Model not supported!"

    return MODEL_PATHS[model_name]


def test_embeddings(model_name: str = "svm", embedding_type = "bert", batch_size: int = None):
    assert embedding_type in EMB_TYPES, "Embedding not supported!"
    data_name = f"test_sliced_stair_twitter_{embedding_type}.h5"
    # data_name = f"hold_out_test_sliced_stair_twitter_{embedding_type}.h5" # TODO This is for demo purposes

    # Load the model from disk
    model_dir = _get_model_path(model_name=model_name, embedding_type=embedding_type)
    model_path = str(model_dir / 'sklearn_model.sav')
    model = pickle.load(open(model_path, 'rb'))

    # Get pickled dataframe with embeddings
    data_path = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings" / data_name

    if not batch_size:
        data = read_h5(data_path)
        X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
        y_true = np.array(data["label"])
        y_pred = model.predict(X)
    else:
        n_total_samples = len(read_h5(data_path, col_name="idx"))
        y_pred = []
        y_true = []
        for i in range(0, n_total_samples, batch_size):
            data = read_h5(data_path, start=i, chunk_size=batch_size)
            X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
            y_pred.extend(model.predict(X))
            y_true.extend(data["label"])

    return y_pred, y_true


click.option = partial(click.option, show_default=True)
@click.command()
@click.option("-m", "--model", type=click.Choice(["svm", "naive_bayes", "xgboost", "knn"]), default="svm", help="Model to select from saved models")
@click.option("-e", "--embedding", type=click.Choice(["glove", "fasttext", "bert"]), default="bert", help="Embedding type")
def main(model, embedding):
    print(f"> Printing metrics for {model} using {embedding} embeddings")
    y_pred, y_true = test_embeddings(model, embedding, batch_size=3)
    threshold = 0.5
    out = [1 if pred > threshold else 0 for pred in y_pred]
    stats = get_metrics(out, y_true)
    print_metrics_comprehensive(stats)

if __name__ == "__main__":
    main()
    