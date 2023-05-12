import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from pathlib import Path
import os
import sys
from csv import QUOTE_NONE


def train(max_len: int):
    base_path = Path(os.path.abspath(__file__)).parents[3] / "dataset_creation" / "data" / "train_test" / "new_preprocessed"
    train_path = base_path / f"train_sliced_stair_twitter_{max_len}_preprocessed.csv"
    val_path = base_path / f"val_sliced_stair_twitter_{max_len}_preprocessed.csv"
    
    train_df = pd.read_csv(train_path, sep="‎", quoting=QUOTE_NONE, engine="python")
    val_df = pd.read_csv(val_path, sep="‎", quoting=QUOTE_NONE, engine="python")

    x_train = train_df["text"].values.astype('U')
    y_train = train_df["label"].values

    x_val = val_df["text"].values.astype('U')
    y_val = val_df["label"].values

    whole_x = pd.concat([train_df, val_df])["text"].values.astype('U')
    print(len(whole_x))

    tfidf = TfidfVectorizer().fit(whole_x)
    tfidf_train = tfidf.transform(x_train).toarray()

    # Train
    print("training")
    nb = GaussianNB()
    nb.fit(tfidf_train, y_train)
    tfidf_train, y_train = None, None

    print("testing")
    tfidf_val = tfidf.transform(x_val).toarray()
    predicted_labels = nb.predict(tfidf_val)

    print(classification_report(y_val, predicted_labels))

if __name__ == "__main__":
    train(256)