import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from pathlib import Path
import os
import sys
from csv import QUOTE_NONE
from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score, fbeta_score
import click
from sklearn.feature_selection import chi2


def train(max_len: int):
    """
    Fit a TFIDF-vectorizer on corpus to extract n-gram features.
    Ngram features are presented in top 10 most distinguishing fashion in descending order.

    Args:
        max_len (int): Max length of each text sequence in rows of dataframe.
    """
    base_path = Path(os.path.abspath(__file__)).parents[4] / "data" / "processed_text" / "train_test" / "preprocessed_glove"
    train_path = base_path / f"train_sliced_stair_twitter_{max_len}_preprocessed.csv"
    val_path = base_path / f"val_sliced_stair_twitter_{max_len}_preprocessed.csv"
    test_path = base_path / f"test_sliced_stair_twitter_{max_len}_preprocessed.csv"

    train_df = pd.read_csv(train_path, sep="‎", quoting=QUOTE_NONE, engine="python")
    val_df = pd.read_csv(val_path, sep="‎", quoting=QUOTE_NONE, engine="python")
    test_df = pd.read_csv(test_path, sep="‎", quoting=QUOTE_NONE, engine="python")

    train_df = pd.concat([train_df, val_df, test_df], axis=0)

    x_train = train_df["text"].values.astype('U')
    y_train = train_df["label"].values

    tfidf = TfidfVectorizer(sublinear_tf=True, norm="l2", ngram_range=(1,1), stop_words="english")
    tfidf_train = tfidf.fit_transform(x_train).toarray()

    # N-gram filters to remove unwanted n-grams specific to individuals
    # These are optional to use
    unwanted_unigrams = ["mackenzie", "rachael", "andrew", "embersghostsquad", "egs", "rt", "mackenziewest", "andrewblaze"]
    unwanted_bigrams = ["egs wiki", "ghost squad", "rachael shadows", "egs embersghostsquad", "fan art", "embersghostsquad egs", "mackenzie west"]

    N = 10
    for label in range(2):
        features_chi2 = chi2(tfidf_train, y_train == label)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names_out())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

        # Removing unwanted n-grams
        # Can skip this step if no filtering is wanted.
        unigrams = [v for v in unigrams if v not in ["nan", "propname"]]
        unigrams = [word for word in unigrams if word not in unwanted_unigrams]
        bigrams = [sent for sent in bigrams if sent not in unwanted_bigrams]

        print(f"Most correlated for {'non-shooter' if label == 0 else 'shooter'}")
        print(f"Most correlated unigrams:")
        [print(unigram) for unigram in unigrams[-N:]]
        print(f"Most correlated bigrams:")
        [print(bigram) for bigram in bigrams[-N:]]

@click.command()
@click.option("-l", "--max_len", type=click.INT, help="Max length of sentence to be allowed. Determines padding and truncation")
def main(max_len: int):
    train(max_len)

if __name__ == "__main__":
    main()