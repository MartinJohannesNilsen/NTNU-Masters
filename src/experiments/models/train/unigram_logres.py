import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pathlib import Path
import os
from csv import QUOTE_NONE
from sklearn.utils import class_weight
from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score, fbeta_score, confusion_matrix, roc_auc_score
import click
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from nltk.stem import WordNetLemmatizer

def get_metrics(predictions, labels) -> dict:
    """Get a selection of performance metrics.

    Args:
        predictions (List[float]): List of predictions.
        labels (List[float]): List of labels.

    Returns:
        dict: Dictionary of performance metrics.
    """
    
    # Gather performance metrics
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    specificity = (tn) / (tn + fp)

    fscore = 2 * (precision * recall) / (precision + recall)
    beta = 0.5
    f05score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    beta = 2
    f2score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    
    try:
        roc_auc = roc_auc_score(labels, predictions)
    except:
        roc_auc = None
    
    # Create dictionary of metrics
    metrics = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": fscore,
        "roc_auc": roc_auc,
        "f2_score": f2score,
        "f_05_score": f05score
    }

    return metrics

def train(max_len: int):
    base_path = Path(os.path.abspath(__file__)).parents[3] / "dataset_creation" / "data" / "train_test" / "new_preprocessed"
    train_path = base_path / f"train_sliced_stair_twitter_{max_len}_preprocessed.csv"
    val_path = base_path / f"val_sliced_stair_twitter_{max_len}_preprocessed.csv"
    
    train_df = pd.read_csv(train_path, sep="‎", quoting=QUOTE_NONE, engine="python")
    val_df = pd.read_csv(val_path, sep="‎", quoting=QUOTE_NONE, engine="python")

    train_df = pd.concat([train_df, val_df], axis=0)


    x_train = train_df["text"].values.astype('U')
    y_train = train_df["label"].values

    tfidf = TfidfVectorizer().fit(x_train)
    tfidf_train = tfidf.transform(x_train).toarray()

    # Train
    print("training")

    class_wts = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(np.array(y_train)),
        y=np.array(y_train)
    )

    wts = {
        0: class_wts[0],
        1: class_wts[1]
    }

    gs_params = {
        'C': [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1","l2"]
    }
    
    scoring = {'recall': make_scorer(recall_score), 'f1': make_scorer(f1_score), 'f2': make_scorer(fbeta_score, beta=2), 'precision': make_scorer(precision_score)}
    cv = StratifiedKFold(n_splits=3)


    logres = LogisticRegression(class_weight=wts)
    grid_search = GridSearchCV(logres, gs_params, scoring=scoring, refit="f2", cv=cv, n_jobs=-1, verbose=1)
    logres = grid_search.fit(tfidf_train, y_train)
    x_train, tfidf_train, y_train = None, None, None
    
    test_path = base_path / f"test_sliced_stair_twitter_{max_len}_preprocessed.csv"
    test_df = pd.read_csv(test_path, sep="‎", quoting=QUOTE_NONE, engine="python")

    print("testing...")

    x_test = test_df["text"].values.astype('U')
    tfidf_test = tfidf.transform(x_test).toarray()
    y_test = test_df["label"].values

    pred_test_labels = logres.predict(tfidf_test)
    print(get_metrics(pred_test_labels, y_test))

@click.command()
@click.option("-l", "--max_len", type=click.INT, help="Max length of sentence to be allowed. Determines padding and truncation")
def main(max_len: int):
    train(max_len)

if __name__ == "__main__":
    main()