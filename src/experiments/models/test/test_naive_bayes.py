import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import sys
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, print_metrics_comprehensive
from experiments.features.load_and_store_emb_batches import read_h5


def test_embeddings(embedding_type = "bert", batch_size: int = None):
    assert embedding_type in ["glove", "fasttext", "bert"], "Embedding not supported!"
    data_name = f"test_sliced_stair_twitter_{embedding_type}.h5"
    # data_name = f"hold_out_test_sliced_stair_twitter_{embedding_type}.h5"

    # Load the model from disk
    model_dir = Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'naive_bayes' / f'{embedding_type}_embeddings'
    model_path = str(model_dir / 'sklearn_model.sav')
    model = pickle.load(open(model_path, 'rb'))

    # Get pickled dataframe with embeddings
    data_path = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings" / data_name

    if not batch_size:
        data = read_h5(data_path)
        X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
        y = np.array(data["label"])
        y_pred = model.predict(X)
    else:
        n_total_samples = len(read_h5(data_path, col_name="idx"))
        y_pred = []
        for i in range(0, n_total_samples, batch_size):
            data = read_h5(data_path, start=i, chunk_size=batch_size)
            X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
            y = np.array(data["label"])
            y_pred.extend(model.predict(X))

    return y_pred, y


if __name__ == "__main__":
    embedding_types = ["glove", "fasttext", "bert"]
    for embedding_type in embedding_types:
        print(embedding_type)
        y_pred, y_true = test_embeddings(embedding_type, batch_size=200)
        threshold = 0.5
        out = [1 if pred > threshold else 0 for pred in y_pred]
        stats = get_metrics(out, y_true)
        print_metrics_comprehensive(stats)