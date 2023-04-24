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
def _get_model_path_embeddings(model_name: str, embedding_type: str, training_file: str):
    assert embedding_type in EMB_TYPES, "Embedding not supported!"

    MODEL_PATHS = {
        "xgboost": Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'xgboost' / f'{embedding_type}_embeddings' / training_file,
        "svm": Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'svm' / f'{embedding_type}_embeddings' / training_file,
        "naive_bayes": Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'naive_bayes' / f'{embedding_type}_embeddings' / training_file,
        "knn": Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'naive_bayes' / f'{embedding_type}_embeddings' / training_file,
    }
    assert model_name in MODEL_PATHS.keys(), "Model not supported!"

    return MODEL_PATHS[model_name]


def test_embeddings(model_name: str = "svm", embedding_type = "bert", batch_size: int = None, training_file: str = ""):
    assert embedding_type in EMB_TYPES, "Embedding not supported!"
    data_name = f"test_sliced_stair_twitter_{embedding_type}.h5"
    # data_name = f"hold_out_test_sliced_stair_twitter_{embedding_type}.h5" # TODO This is for demo purposes

    # Load the model from disk
    model_dir = _get_model_path_embeddings(model_name=model_name, embedding_type=embedding_type, training_file=training_file)
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

LIWC_DICTS = ["2022", "2015", "2007", "2001"]
def _get_model_path_liwc(model_name: str, liwc_dict: str, training_file: str):
    assert liwc_dict in LIWC_DICTS, "LIWC dictionary version not supported!"

    MODEL_PATHS = {
        "xgboost": Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'xgboost' / f'liwc_{liwc_dict}' / training_file,
        "svm": Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'svm' / f'liwc_{liwc_dict}' / training_file,
        "naive_bayes": Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'naive_bayes' / f'liwc_{liwc_dict}' / training_file,
        "knn": Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'naive_bayes' / f'liwc_{liwc_dict}' / training_file,
    }
    assert model_name in MODEL_PATHS.keys(), "Model not supported!"

    return MODEL_PATHS[model_name]


def test_liwc(model_name: str = "svm", liwc_dict = "2022", batch_size: int = None, training_file: str = ""):
    assert liwc_dict in LIWC_DICTS, "LIWC dictionary version not supported!"
    # data_name = f"LIWC-22 Results - train_sliced_stair_twitter - LIWC Analysis.h5"
    data_name = f"LIWC-22 Results - shooter_hold_out_test - LIWC Analysis.h5"  # TODO This is for demo purposes

    # Load the model from disk
    model_dir = _get_model_path_liwc(model_name=model_name, liwc_dict=liwc_dict, training_file=training_file)
    model_path = str(model_dir / 'sklearn_model.sav')
    model = pickle.load(open(model_path, 'rb'))

    # Get pickled dataframe with embeddings
    data_path = Path(os.path.abspath(__file__)).parents[2] / "features" / "liwc" / liwc_dict / data_name

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
@click.option("-e", "--embedding", type=click.Choice(["glove", "fasttext", "bert"]), default=None, help="Embedding type")
@click.option("-l", "--liwc", type=click.Choice(["2022", "2015", "2007", "2001"]), default=None, help="LIWC dictionary")
@click.option("-t", "--training-file", help="Name of file used for training")
def main(model, embedding, liwc, training_file):
    assert embedding or liwc, "Please select feature set!"
    if embedding:
        assert os.path.isDir(str(Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / model / "embeddings" / embedding / training_file)), "Model not found"
        print(f"> Printing metrics for {model} using {embedding} embeddings")
        y_pred, y_true = test_embeddings(model, embedding, batch_size=3, training_file=training_file)
        threshold = 0.5
        out = [1 if pred > threshold else 0 for pred in y_pred]
        stats = get_metrics(out, y_true)
        print_metrics_comprehensive(stats)
    if liwc:
        assert os.path.isDir(str(Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / model / "LIWC" / liwc / training_file)), "Model not found"
        print(f"> Printing metrics for {model} using the {liwc} dictionary")
        y_pred, y_true = test_liwc(model, liwc, batch_size=3, training_file=training_file)
        threshold = 0.5
        out = [1 if pred > threshold else 0 for pred in y_pred]
        stats = get_metrics(out, y_true)
        print_metrics_comprehensive(stats)

if __name__ == "__main__":
    main()
    
    # Example liwc
    # python test_sklearn.py -m svm -l 2022 -t "LIWC-22 Results - shooter_hold_out_test - LIWC Analysis"
    
    # Example embeddings
    # python test_sklearn.py -m svm -e glove -t "test_sliced_stair_twitter_glove_50_tail_256"
    