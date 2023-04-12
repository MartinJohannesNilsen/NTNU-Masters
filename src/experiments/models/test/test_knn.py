import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import sys
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, print_metrics_comprehensive


def test_embeddings(embedding_type = "bert"):
    assert embedding_type in ["glove", "fasttext", "bert"], "Embedding not supported!"
    pickled_df_name = f"DEMO_test_no_stair_twitter_{embedding_type}.pkl"

    # Load the model from disk
    model_dir = Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'knn' / f'{embedding_type}_embeddings'
    model_path = str(model_dir / 'sklearn_model.sav')
    model = pickle.load(open(model_path, 'rb'))

    # Get pickled dataframe with embeddings
    data_path = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings" / pickled_df_name
    df = pd.read_pickle(data_path, compression="bz2")
    X = np.array([element.ravel().tolist() for element in df["text"].values])
    y = np.array(df["label"].values)

    return model.predict(X), y


if __name__ == "__main__":
    embedding_type = "bert" # "glove", "fasttext", "bert"
    y_pred, y_true = test_embeddings(embedding_type)
    stats = get_metrics(y_pred, y_true)
    print_metrics_comprehensive(stats)