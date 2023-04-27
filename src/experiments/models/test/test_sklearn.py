from csv import QUOTE_NONE
from functools import partial
import os
from pathlib import Path
from typing import List
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

def _get_data_path_from_emb_path(emb_path: str):
    dataset_dir = Path(os.path.abspath(__file__)).parents[3] / "dataset_creation" / "data" / "train_test"
    purpose = "shooter_hold_out_test" if "hold_out_test" in emb_path else "test" if "test" in emb_path else "train"
    size = "_256" if "256" in emb_path else ""
    stair_twitter = "_no_stair_twitter" if "no_stair_twitter" in emb_path else "_sliced_stair_twitter" if "sliced_stair_twitter" in emb_path else ""
    file_path = str(dataset_dir / (purpose + stair_twitter + size + ".csv"))
    assert os.path.isfile(file_path), "File not found!"
    return file_path


def get_texts_matching_tensors(emb_features_path, matching_dataset_path) -> List[str]:
    indexes = read_h5(emb_features_path, col_name="idx")
    dataset = pd.read_csv(matching_dataset_path, sep="‎", quoting=QUOTE_NONE, engine="python")
    filtered_df = dataset.loc[dataset.index[indexes]]
    assert len(filtered_df.index) == len(indexes), "Could not align h5 indices with original dataset!"
    return filtered_df["text"].to_list()


def get_texts_liwc(liwc_features_path) -> List[str]:
    # As liwc features do not filter out, we can return matching dataset path. Just ensure same length
    features = pd.read_csv(liwc_features_path)
    return features["text"].to_list()

def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))

click.option = partial(click.option, show_default=True)
@click.command()
@click.argument("model_path", nargs=1)
@click.argument("test_path", nargs=1)
@click.option("-t", "--threshold", type=float, default=0.5, help="Threshold for predictions")
@click.option("-o", "--output", type=str, help="Output posts grouped by confusion matrix to file")
def main(model_path, test_path, threshold, output):
    assert threshold >= 0 and threshold <= 1, "Threshold needs to be between 0 and 1!"

    print(f"Threshold = {threshold}")

    # Check that paths leads to files
    assert os.path.isfile(model_path), "No model file found!"
    assert os.path.isfile(test_path), "No test file found!"

    if "embeddings" in model_path:
        assert "embeddings" in test_path, "Mismatch between model and test embeddings path!"
        y_pred, y_true = test_embeddings(model_path, test_path)
        y_pred_sigmoid = [sigmoid_function(pred) for pred in y_pred]
        out = [1 if pred > threshold else 0 for pred in y_pred_sigmoid]
        stats = get_metrics(out, y_true)
        print_metrics_comprehensive(stats)
        if output:
            # Get texts ordered by confusion matrix
            emb_text_path = _get_data_path_from_emb_path(test_path)
            emb_texts = get_texts_matching_tensors(test_path, emb_text_path)
            post_dict = get_posts_ordered_by_confusion_matrix(emb_texts, out, y_true)
            
            # Write to output file
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, "w") as f:
                f.write(f"{'%'*20}\nThreshold = {threshold}\n{'%'*20}\n")
                for k, v in post_dict.items():
                    f.write(f"{'%'*10}\n{'%'*2}  {k}  {'%'*2}\n{'%'*10}\n")
                    f.writelines(line + "\n" for line in v)
                    f.writelines("\n")
    
    elif "liwc" in model_path:
        assert "liwc" in test_path, "Mismatch between model and test LIWC dictionaries path!"
        y_pred, y_true = test_liwc(model_path, test_path)
        y_pred_sigmoid = [sigmoid_function(pred) for pred in y_pred]
        # for pred in y_pred:
        #     if pred > 1 or pred < 0:
        #         print(pred)
        out = [1 if pred > threshold else 0 for pred in y_pred_sigmoid]
        stats = get_metrics(out, y_true)
        print_metrics_comprehensive(stats)
        if output:
            # Get texts ordered by confusion matrix
            liwc_text_path = test_path.replace("h5", "csv")
            liwc_texts = get_texts_liwc(liwc_text_path)
            post_dict = get_posts_ordered_by_confusion_matrix(liwc_texts, out, y_true)
            
            # Write to output file
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, "w") as f:
                f.write(f"{'%'*20}\nThreshold = {threshold}\n{'%'*20}\n")
                for k, v in post_dict.items():
                    f.write(f"{'%'*10}\n{'%'*2}  {k}  {'%'*2}\n{'%'*10}\n")
                    f.writelines(line + "\n" for line in v)
                    f.writelines("\n")



if __name__ == "__main__":
    main()

    # emb = "/Users/martinjohannesnilsen/NTNU/Datateknologi/4. semester/Master's Thesis/Source Code/src/experiments/features/embeddings/test_sliced_stair_twitter_glove_50_tail_256.h5"
    # dataset = "/Users/martinjohannesnilsen/NTNU/Datateknologi/4. semester/Master's Thesis/Source Code/src/dataset_creation/data/train_test/test_sliced_stair_twitter_256.csv"
    # get_texts_matching_tensors(emb, dataset)
    
    # liwc = "/Users/martinjohannesnilsen/NTNU/Datateknologi/4. semester/Master's Thesis/Source Code/src/experiments/features/liwc/csv/2022/LIWC-22 Results - test_sliced_stair_twitter - LIWC Analysis.csv"
    # print(get_texts_matching_liwc(liwc))

    # python test_sklearn.py "/Users/martinjohannesnilsen/NTNU/Datateknologi/4. semester/Master's Thesis/Source Code/src/experiments/models/saved_models/svm/embeddings/glove/test_sliced_stair_twitter_glove_50_tail_256/sklearn_model.sav" "/Users/martinjohannesnilsen/NTNU/Datateknologi/4. semester/Master's Thesis/Source Code/src/experiments/features/embeddings/test_sliced_stair_twitter_glove_50_tail_256.h5" --output "./out.txt"