
import numpy as np
import pandas as pd
from csv import QUOTE_NONE
import sys
import csv
from pathlib import Path
import os
import pickle   
from sklearn.model_selection import KFold
from tabulate import tabulate
import sys
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, get_average_metrics
from sklearn.gaussian_process import kernels, GaussianProcessClassifier


def get_data(embedding_type = "glove", data_type = "train"):
    assert embedding_type in ["glove", "fasttext", "bert"], "Embedding not supported!"

    # Get pickled dataframe with embeddings
    data_path = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings" / f"{data_type}_sliced_stair_twitter_{embedding_type}.pkl"
    df = pd.read_pickle(data_path, compression="bz2")

    # Train data inputs X and labels y
    X = np.array([element.ravel().tolist() for element in df["text"].values]) # Flatten (512, emb_dim) into (512*emb_dim) with ravel, make list and output a numpy array
    y = np.array(df["label"].values)

    return X, y

# Brute force test kernels
# Change later if better understanding

n = 50

kns = {"rbf": kernels.RBF(),
       "rq": kernels.RationalQuadratic(),
       "white": kernels.WhiteKernel()}

x_train, y_train = get_data(embedding_type="glove", data_type="train")
x_test, y_test = get_data(embedding_type="glove", data_type="test")


all_metrics = []
for k_name, k in kns.items():

    print(f"Training for {k_name}")

    gp = GaussianProcessClassifier(kernel=k, n_restarts_optimizer=5)
    gp.fit(x_train, y_train)

    y_pred = gp.predict(x_test)

    kernel_metrics = [k_name]
    for metric in get_metrics(y_pred, y_test).values():
        kernel_metrics.append(metric)
    print(kernel_metrics)
    all_metrics.append(kernel_metrics)


print(tabulate(all_metrics, headers=["Kernel name", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC"]))