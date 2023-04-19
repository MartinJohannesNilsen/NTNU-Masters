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


def test_embeddings(embedding_type = "bert"):
    assert embedding_type in ["glove", "fasttext", "bert"], "Embedding not supported!"
    # data_name = f"test_sliced_stair_twitter_{embedding_type}.h5"
    data_name = f"hold_out_test_sliced_stair_twitter_{embedding_type}.h5"

    # Load the model from disk
    model_dir = Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'xgboost' / f'{embedding_type}_embeddings'
    model_path = str(model_dir / 'sklearn_model.sav')
    model = pickle.load(open(model_path, 'rb'))

    # Get pickled dataframe with embeddings
    data_path = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings" / data_name
    data = read_h5(data_path)
    X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
    y = np.array(data["label"])

    return model.predict(X), y


if __name__ == "__main__":
    embedding_type = "bert" # "glove", "fasttext", "bert"
    print(embedding_type)
    y_pred, y_true = test_embeddings(embedding_type)
    threshold = 0.75
    out = [1 if pred > threshold else 0 for pred in y_pred]
    stats = get_metrics(out, y_true)
    print_metrics_comprehensive(stats)