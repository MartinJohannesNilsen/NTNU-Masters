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
from experiments.utils.metrics import get_metrics, print_metrics_comprehensive, get_posts_ordered_by_confusion_matrix, print_metrics_tabulated, print_metrics_simplified
from experiments.utils.create_and_store_embs import read_h5


def test_embeddings(model_path, test_path, batch_size: int = None):
    """Method for testing sklearn models on word embeddings.

    Args:
        model_path (str): Path to trained model.
        test_path (str): Path to test dataset.
        batch_size (int, optional): Optional batch size for iterative testing. Defaults to None.

    Returns:
        (List, List, List, List): Lists of predicted labels, predicted scores, labels and indices.
        
    """

    # Load the model from disk
    model = pickle.load(open(model_path, 'rb'))

    if not batch_size:
        # data = read_h5(test_path)
        data = read_h5(test_path, chunk_size=10)
        X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
        y_true = np.array(data["label"])

        y_pred = list(model.predict(X))
        y_score = [prob[1] for prob in model.predict_proba(X)]
        # y_score_pred = [np.argmax(prob) for prob in model.predict_proba(X)] # Same as y_pred
        # y_confidence = [max(prob) for prob in model.predict_proba(X)]
        idxs = np.array(data["idx"])
        
    else:
        n_total_samples = len(read_h5(test_path, col_name="idx"))
        y_pred = []
        y_score = []
        y_true = []
        idxs = []
        for i in range(0, n_total_samples, batch_size):
            data = read_h5(test_path, start=i, chunk_size=batch_size)
            X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
            y_pred.extend(model.predict(X))
            y_score.extend([prob[1] for prob in model.predict_proba(X)])
            y_true.extend(data["label"])
            idxs.extend(data["idx"])

    return y_pred, y_score, y_true, idxs


def test_liwc(model_path, test_path, batch_size: int = None):
    """Method for testing sklearn models on LIWC features.

    Args:
        model_path (str): Path to trained model.
        test_path (str): Path to test dataset.
        batch_size (int, optional): Optional batch size for iterative testing. Defaults to None.

    Returns:
        (List, List, List, List): Lists of predicted labels, predicted scores, labels and indices.
        
    """

    # Load the model from disk
    model = pickle.load(open(model_path, 'rb'))

    if not batch_size:
        data = read_h5(test_path)
        X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
        y_true = np.array(data["label"])

        y_pred = list(model.predict(X))
        y_score = [prob[1] for prob in model.predict_proba(X)]
        idxs = np.array(data["idx"])

    else:
        n_total_samples = len(read_h5(test_path, col_name="idx"))
        y_pred = []
        y_score = []
        y_true = []
        idxs = []
        for i in range(0, n_total_samples, batch_size):
            data = read_h5(test_path, start=i, chunk_size=batch_size)
            X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
            y_pred.extend(model.predict(X))
            y_score.extend([prob[1] for prob in model.predict_proba(X)])
            y_true.extend(data["label"])
            idxs.extend(data["idx"])

    return y_pred, y_score, y_true, idxs

def _get_data_path_from_emb_path(emb_path: str):
    dataset_dir = Path(os.path.abspath(__file__)).parents[4] / "data" / "processed_data" / "train_test" / "csv"
    purpose = "shooter_hold_out" if "shooter_hold_out" in emb_path else "test" if "test" in emb_path else "train"
    size = "_256" if "256" in emb_path else "_512"
    stair_twitter = "_no_stair_twitter" if "no_stair_twitter" in emb_path else "_sliced_stair_twitter" if "sliced_stair_twitter" in emb_path else ""
    file_path = str(dataset_dir / (purpose + stair_twitter + size + ".csv"))
    assert os.path.isfile(file_path), "File not found!"
    return file_path


def get_texts_matching_tensors(emb_features_path, matching_dataset_path) -> List[str]:
    # indexes = read_h5(emb_features_path, col_name="idx")
    indexes = read_h5(emb_features_path, col_name="idx", chunk_size=10)
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
@click.option("-l", "--use-list-of-thresholds", is_flag=True, help="Iterate over list of thresholds")
@click.option("-o", "--output", type=str, help="Output posts grouped by confusion matrix to file")
def main(model_path, test_path, threshold, use_list_of_thresholds, output):
    assert threshold >= 0 and threshold <= 1, "Threshold needs to be between 0 and 1!"
    list_of_thresholds = list(range(0, 1, 0.05)) if use_list_of_thresholds else None

    # Check that paths leads to files
    assert os.path.isfile(model_path), "No model file found!"
    assert os.path.isfile(test_path), "No test file found!"

    def test(t):

        if "embeddings" in model_path:
            assert "embeddings" in test_path, "Mismatch between model and test embeddings path!"
            pred_labels, pred_scores, y_true, idxs = test_embeddings(model_path, test_path)
            # pred_labels = [1 if pred > t else 0 for pred in pred_scores]
            metrics = get_metrics(pred_labels, y_true)
        
        elif "liwc" in model_path:
            assert "liwc" in test_path, "Mismatch between model and test LIWC dictionaries path!"
            pred_labels, pred_scores, y_true, idxs = test_liwc(model_path, test_path)
            # pred_labels = [1 if pred > t else 0 for pred in pred_scores]
            metrics = get_metrics(pred_labels, y_true)
        
        return metrics, pred_labels, pred_scores, y_true, idxs


    def uniquify(path):
        filename, extension = os.path.splitext(path)
        counter = 1

        while os.path.exists(path):
            # path = filename + " (" + str(counter) + ")" + extension
            path = filename + "_" + str(counter) + extension
            counter += 1

        return path

    def write_output(preds, labels, t, idxs = None, scores = None):
        """Function for writing output."""
        
        # Write texts grouped by confusion matrix (TP, TN, FP, FN)
        """  
        # Get texts ordered by confusion matrix
        emb_text_path = _get_data_path_from_emb_path(test_path)
        emb_texts = get_texts_matching_tensors(test_path, emb_text_path)
        post_dict = get_posts_ordered_by_confusion_matrix(emb_texts, preds, labels)
        
        # Write to output file
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(uniquify(output), "w") as f:
            f.write(f"{'%'*20}\nThreshold = {t}\n{'%'*20}\n")
            for k, v in post_dict.items():
                f.write(f"{'%'*10}\n{'%'*2}  {k}  {'%'*2}\n{'%'*10}\n")
                f.writelines(line + "\n" for line in v)
                f.writelines("\n")
        """
        
        if idxs is not None and scores is not None:
            os.makedirs(os.path.dirname(output.replace("posts", "scores")), exist_ok=True)
            with open(uniquify(output.replace("posts", "scores")), "w") as f:
                f.write("idx,pred_val,pred_label,label\n")
                
                for i, score, pred, label in zip(idxs, scores, preds, labels):
                    f.write(f"{i},{score},{pred},{label}\n")

    if list_of_thresholds:
        list_of_metrics = []
        for t in list_of_thresholds:
            metrics, pred_labels, pred_scores, y_true, idxs = test(t)
            list_of_metrics.append(metrics)
            if output:
                write_output(pred_labels, y_true, t, idxs, pred_scores)
        print_metrics_tabulated(keys=list_of_thresholds, list_of_metrics=list_of_metrics)

    else:
        metrics, pred_labels, pred_scores, y_true, idxs = test(threshold)
        # print_metrics_comprehensive(metrics)
        print_metrics_simplified(metrics)
        if output:
            write_output(pred_labels, y_true, threshold, idxs, pred_scores)


if __name__ == "__main__":
    main()