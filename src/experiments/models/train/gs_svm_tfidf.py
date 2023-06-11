import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
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
print(Path(os.path.abspath(__file__)).parents[2])
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
from utils.metrics import get_metrics
import pickle
import click


def _save_model(model, saved_model_dir, name = 'sklearn_model.sav'):
    # Save the model to disk
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(saved_model_dir / name)
    pickle.dump(model, open(model_path, 'wb'))
    
    return model_path

def train(max_len: int):
    """
    Gridsearch svm model on tfidf features
    Save best performing model to file.
    Param space is searched based on f2 score for each config.

    Args:
        max_len (int): Max length of text sequence for each row in dataframe.
    """
    
    base_path = Path(os.path.abspath(__file__)).parents[4] / "data" / "processed_data" / "train_test" / "preprocessed_glove"
    train_path = base_path / f"train_sliced_stair_twitter_{max_len}_preprocessed.csv"
    val_path = base_path / f"val_sliced_stair_twitter_{max_len}_preprocessed.csv"
    
    train_df = pd.read_csv(train_path, sep="‎", quoting=QUOTE_NONE, engine="python")
    val_df = pd.read_csv(val_path, sep="‎", quoting=QUOTE_NONE, engine="python")

    train_df = pd.concat([train_df, val_df], axis=0)

    x_train = train_df["text"].values.astype('U')
    y_train = train_df["label"].values

    tfidf = TfidfVectorizer().fit(x_train)
    tfidf_train = tfidf.transform(x_train).toarray()

    scoring = {'recall': make_scorer(recall_score), 'f1': make_scorer(f1_score), 'f2': make_scorer(fbeta_score, beta=2), 'precision': make_scorer(precision_score)}

    # Train
    print("training")
    svm = SVC()

    # Gridsearch parameter space
    gs_params = {
        'C': [1, 10, 100],
        'kernel': ['linear'],
        'gamma': ['scale', 'auto']
        }

    cv = StratifiedKFold(n_splits=2)
    grid_search = GridSearchCV(svm, gs_params, scoring=scoring, refit="f2", cv=cv, n_jobs=-1, verbose=1)
    print("gridsearch")
    svm = grid_search.fit(tfidf_train, y_train)
    x_train, tfidf_train, y_train = None, None, None
    
    test_path = base_path / f"test_sliced_stair_twitter_{max_len}_preprocessed.csv"
    test_df = pd.read_csv(test_path, sep="‎", quoting=QUOTE_NONE, engine="python")

    print("testing...")

    x_test = test_df["text"].values.astype('U')
    tfidf_test = tfidf.transform(x_test).toarray()
    y_test = test_df["label"].values

    pred_test_labels = svm.predict(tfidf_test)
    ms = get_metrics(y_test, pred_test_labels)
    print(ms)

    print("Best params:")
    print(svm.best_params_)

    saved_path = pickle.dumps(svm, Path(__file__).parents[1] / "saved_models", name=f"svm_tfidf_sklearn_{max_len}_check_linear.sav")

    print(svm.coef_)
    pickle.dumps(svm.coef_, str(Path(__file__).parents[1] / "results" / "coef_weights"))

@click.command()
@click.option("-l", "--max_len", type=click.INT, help="Max length of sentence to be allowed. Determines padding and truncation")
def main(max_len: int):
    train(max_len)

if __name__ == "__main__":
    main()