from functools import partial
import os
from pathlib import Path
import click
import numpy as np
import time
import pickle
from sklearn.metrics import make_scorer, recall_score   
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, train_test_split
import joblib
from tabulate import tabulate
import sys
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, get_average_metrics
from experiments.features.load_and_store_emb_batches import read_h5
# Models
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.gaussian_process import kernels, GaussianProcessClassifier
from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score

def combined_recall_f1(y_true, y_pred, recall_weight=0.5):
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return recall_weight * recall + (1 - recall_weight) * f1

"""
grid_search_params = {
    "svm" : (SVC(), {
        'C': np.logspace(-3, 3, 7),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto']}),
    "sgd": (SGDClassifier(), {
        'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty': ['none', 'l1', 'l2', 'elasticnet'],
        'alpha': np.logspace(-6, 3, 10),
        'l1_ratio': np.linspace(0, 1, 11),
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': np.logspace(-4, 0, 5)}),
    "nb" : (GaussianNB(), {}),
    "knn": (KNeighborsClassifier(), {
        'n_neighbors': range(1, 21),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']}),
    "xgboost": (XGBClassifier(eval_metric='logloss'), {
        'n_estimators': range(50, 301, 50),
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': range(3, 11),
        'min_child_weight': range(1, 7),
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5]}),
    "gaussian": (GaussianProcessClassifier(), {
        'kernel': [kernels.RBF(), kernels.DotProduct(), kernels.Matern(), kernels.WhiteKernel()],
        'optimizer': ['fmin_l_bfgs_b', 'fmin_cg'],
        'n_restarts_optimizer': [0, 1, 2, 3, 4],
        'max_iter_predict': [50, 100, 150, 200]})
}
"""
grid_search_params = {
    "svm" : (SVC(), {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'sigmoid'],
        # 'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto']}),
    "sgd": (SGDClassifier(loss="hinge"), {
        # 'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty': ['none', 'l2', 'elasticnet'],
        'alpha': np.logspace(-6, 3, 10),
        # 'l1_ratio': np.linspace(0, 1, 11),
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        # 'eta0': np.logspace(-4, 0, 5)
        }),
    "nb" : (GaussianNB(), {}),
    "knn": (KNeighborsClassifier(), {
        'n_neighbors': range(1, 21),
        # 'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']}),
    "xgboost": (XGBClassifier(eval_metric='logloss'), {
        'n_estimators': range(50, 301, 50),
        'learning_rate': [0.01, 0.025, 0.05, 0.1],
        # 'max_depth': range(3, 11),
        # 'min_child_weight': range(1, 7),
        # 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        #'gamma': [0, 1, 10]
        }),
    "gaussian": (GaussianProcessClassifier(), {
        # 'kernel': [kernels.RBF(), kernels.DotProduct(), kernels.WhiteKernel()],
        'kernel': [kernels.DotProduct() + kernels.WhiteKernel(), kernels.RBF(length_scale=1.0) + kernels.WhiteKernel()],
        # 'optimizer': ['fmin_l_bfgs_b', 'fmin_cg'],
        #'n_restarts_optimizer': [0, 1, 2, 3, 4],
        # 'max_iter_predict': [50, 100, 150, 200]
        })
}


# Training utitilites
def _get_model(model = "xgboost"):
    if model == "svm":
        return SVC(probability=True)
    if model == "sgd":
        return SGDClassifier()
    elif model == "nb":
        return GaussianNB()
    elif model == "knn":
        return KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance') # n_neighbors=3 if sliced, 5 if no
    elif model == "xgboost":
        return XGBClassifier(eval_metric="logloss")
    elif model == "gaussian":
        return GaussianProcessClassifier()
    else:
        raise NotImplementedError()

def _save_model(model, saved_model_dir, name = 'sklearn_model.sav'):
    # Save the model to disk
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(saved_model_dir / name)
    pickle.dump(model, open(model_path, 'wb'))
    
    return model_path

# Training
def training(saved_model_dir, path, model_type, batch_size = None, grid_search_metric = "f1", random_search = False):

    # Get model
    model = _get_model(model_type)
    
    # Run cross_val if number of splits is defined
    if grid_search_metric:

        # Get classifier and grid params
        print(f"Hyperparameter tuning using sklearn {'RandomizedSearchCV' if random_search else 'GridSearchCV'}")
        classifier, grid_params = grid_search_params[model_type]
        
        # Read data
        print("Loading features and labels ...")
        start_time = time.time()
        features = read_h5(path, keep_tensor_as_ndarray=True, col_name="emb_tensor")
        labels = read_h5(path, col_name="label")
        print(f"Finished loading in {round(time.time() - start_time, 0)} seconds")

        # Train data inputs X and labels y
        print("Creating X for sklearn model ...")
        start_time = time.time()
        X = np.array([row.ravel() for row in features]) # Flatten (512, emb_dim) into (512*emb_dim) with ravel, make list and output a numpy array
        print(f"Created X in {round(time.time() - start_time, 0)} seconds. Size: {round(X.nbytes / 10**9, 2)}GB")
        y = np.array(labels)

        # Split dataaset into 60% sample
        sample_size = 0.6 if "liwc" in str(saved_model_dir) else 0.25
        print(f"Sample {sample_size*100}% for search ...")
        start_time = time.time()
        X_sample, _, y_sample, _ = train_test_split(X, y, test_size=(1-sample_size), random_state=42, stratify=y)
        print(f"Created sample in {round(time.time() - start_time, 0)} seconds. Size: {round(X_sample.nbytes / 10**9, 2)}GB")

        # Set up cross-validation
        cv = StratifiedKFold(n_splits=3)

        # Define scoring metrics
        scoring = {'recall': make_scorer(recall_score), 'f1': make_scorer(f1_score), 'recall_f1': make_scorer(combined_recall_f1, greater_is_better = True), 'precision': make_scorer(precision_score)}

        # Run search
        print(f"Running hyperparameter search ...")
        start_time = time.time()
        verbosity = 1
        # memory = joblib.Memory(location=str(Path(os.path.abspath(__file__)).parents[4] / 'resources' / ".sklearn_cache"), verbose=0)
        if "liwc" in str(saved_model_dir):
            if random_search:
                grid_search = RandomizedSearchCV(classifier, grid_params, scoring=scoring, refit=grid_search_metric, cv=cv, n_jobs=-1, random_state=42, n_iter=10, verbose=verbosity)
            else:
                grid_search = GridSearchCV(classifier, grid_params, scoring=scoring, refit=grid_search_metric, cv=cv, n_jobs=-1, verbose=verbosity)
        else:
            if random_search:
                grid_search = RandomizedSearchCV(classifier, grid_params, scoring=scoring, refit=grid_search_metric, cv=cv, random_state=42, n_iter=10, n_jobs=1, verbose=verbosity) # pre_dispatch=1, memory=memory
            else:
                grid_search = GridSearchCV(classifier, grid_params, scoring=scoring, refit=grid_search_metric, cv=cv, n_jobs=1, verbose=verbosity)
        grid_search.fit(X_sample, y_sample)
        print(f"Finished search in {round(time.time() - start_time, 0)} seconds")

        # Get the index of the best combination of parameters
        best_index = grid_search.best_index_

        # Print the best scores for both metrics
        print("=" * 80)
        print("Optimized metric:", grid_search_metric)
        print("-" * 80)
        print("Best F1 score:", grid_search.cv_results_['mean_test_f1'][best_index])
        print("Best Recall score:", grid_search.cv_results_['mean_test_recall'][best_index])
        print("Best Combined Recall-F1 score:", grid_search.cv_results_['mean_test_recall_f1'][best_index])
        print("Best Precision score:", grid_search.cv_results_['mean_test_precision'][best_index])
        print("-" * 80)
        print("Best params:", grid_search.best_params_)
        print("Best estimator:", grid_search.best_estimator_)
        print("=" * 80)

    # If regular training
    else:
        if batch_size:
            print(f"Training with batch size {batch_size}")
            n_samples_total = len(read_h5(path, col_name="idx"))
            # Iterate over all batches
            for i in range(0, n_samples_total, batch_size):
            
                # Read data
                print(f"Loading features and labels for batch {i} ...")
                start_time = time.time()
                features = read_h5(path, start=i, chunk_size=batch_size, keep_tensor_as_ndarray=True, col_name="emb_tensor")
                labels = read_h5(path, start=i, chunk_size=batch_size, col_name="label")
                print(f"Finished loading in {round(time.time() - start_time, 0)} seconds")

                # Train data inputs X and labels y
                print("Creating X for sklearn model ...")
                start_time = time.time()
                X = np.array([row.ravel() for row in features]) # Flatten (512, emb_dim) into (512*emb_dim) with ravel, make list and output a numpy array
                print(f"Created X in {round(time.time() - start_time, 0)} seconds. Size: {round(X.nbytes / 10**9, 2)}GB")
                y = np.array(labels)

                # Fit model
                print("Fitting model to data ...")
                start_time = time.time()
                model.partial_fit(X, y)
                print(f"Fitted model to data in {round(time.time() - start_time, 0)} seconds")
        
        else:
            print("Training without batch size")
            # Read data
            print("Loading features and labels ...")
            start_time = time.time()
            features = read_h5(path, keep_tensor_as_ndarray=True, col_name="emb_tensor")
            labels = read_h5(path, col_name="label")
            print(f"Finished loading in {round(time.time() - start_time, 0)} seconds")

            # Train data inputs X and labels y
            print("Creating X for sklearn model ...")
            start_time = time.time()
            X = np.array([row.ravel() for row in features]) # Flatten (512, emb_dim) into (512*emb_dim) with ravel, make list and output a numpy array
            print(f"Created X in {round(time.time() - start_time, 0)} seconds. Size: {round(X.nbytes / 10**9, 2)}GB")
            y = np.array(labels)

            # Fit model
            print("Fitting model to data ...")
            start_time = time.time()
            model.fit(X, y)
            print(f"Fitted model to data in {round(time.time() - start_time, 0)} seconds")
            
        # Save model
        _save_model(model, saved_model_dir=saved_model_dir)
        

# Training based on selected feature
SUPPORTED_EMBEDDINGS = ["glove", "glove_50", "fasttext", "bert"]
def train_embeddings(path: str, emb:str, model:str, batch_size = None, grid_search_metric = None):
    assert emb in SUPPORTED_EMBEDDINGS, "Embedding not supported!"
    training(saved_model_dir=Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / model / 'embeddings' / emb / Path(path).stem, path=path, model_type=model, batch_size=batch_size, grid_search_metric=grid_search_metric)

SUPPORTED_LIWC_DICTS = ["2022", "2015", "2007", "2001"]
def train_liwc(path: str, liwc_dict:str, model:str, batch_size = None, grid_search_metric = None):
    assert liwc_dict in SUPPORTED_LIWC_DICTS, "Liwc dictionary not supported!"
    training(saved_model_dir=Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / model / 'liwc' / liwc_dict / Path(path).stem, path=path, model_type=model, batch_size=batch_size, grid_search_metric=grid_search_metric)


click.option = partial(click.option, show_default=True)
@click.command()
@click.argument("path", nargs=1)
@click.option("-m", "--model", type=click.Choice(["svm", "sgd", "nb", "xgboost", "knn", "gaussian"]), default="svm", help="Model to select from saved models")
@click.option("-g", "--grid-search-metric", type=click.Choice(["f1", "recall", "recall_f1", "precision", None]), default=None, help="Perform grid search with given metric")
def main(path, model, grid_search_metric):

    # Check that path leads to file
    assert os.path.isfile(path), "No file found!"

    if "embeddings" in path:
        # Find emb_type
        emb_type = None
        if "glove_50" in path:
            emb_type = "glove_50"
        elif "glove" in path:
            emb_type = "glove"
        elif "fasttext" in path:
            emb_type = "fasttext"
        elif "bert" in path:
            emb_type = "bert"
        assert emb_type, "Incorrect format, could not find embedding!"
        train_embeddings(path, emb_type, model, grid_search_metric = grid_search_metric)
        

    elif "liwc" in path:
        # Find liwc_dict
        liwc_dict = None
        if "2022" in path:
            liwc_dict = "2022"
        elif "2015" in path:
            liwc_dict = "2015"
        elif "2007" in path:
            liwc_dict = "2007"
        elif "2001" in path:
            liwc_dict = "2001"
        assert liwc_dict, "Incorrect format, could not find LIWC dict!"
        train_liwc(path, liwc_dict, model, grid_search_metric = grid_search_metric)

if __name__ == "__main__":
    main()