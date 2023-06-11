from functools import partial
import os
from pathlib import Path
import click
import numpy as np
import time
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, train_test_split
import joblib
from tabulate import tabulate
import sys
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, get_average_metrics
from experiments.utils.create_and_store_embs import read_h5
# Models
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score, fbeta_score

training_params_from_gridsearch = {'nb_256_bert_768_split': {}, 'nb_256_bert_768_tail': {}, 'nb_256_glove_300_split': {}, 'nb_256_bert_768_head': {}, 'nb_256_fasttext_300_split': {}, 'nb_256_glove_300_tail': {}, 'nb_256_glove_50_head': {}, 'nb_256_fasttext_300_tail': {}, 'nb_256_glove_50_split': {}, 'nb_256_glove_50_tail': {}, 'nb_256_glove_300_head': {}, 'nb_256_fasttext_300_head': {}, 'knn_512_fasttext_300_split': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_512_fasttext_300_head': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_512_glove_300_head': {'metric': 'euclidean', 'n_neighbors': 1}, 'knn_512_glove_300_split': {'metric': 'euclidean', 'n_neighbors': 1}, 'knn_512_bert_768_tail': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_512_bert_768_head': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_512_glove_50_split': {'metric': 'euclidean', 'n_neighbors': 1}, 'knn_512_glove_300_tail': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_512_fasttext_300_tail': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_512_glove_50_head': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_512_bert_768_split': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_512_glove_50_tail': {'metric': 'euclidean', 'n_neighbors': 1}, 'svm_512_glove_300_head': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_512_bert_768_head': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}, 'svm_512_fasttext_300_head': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_512_glove_300_tail': {'C': 10, 'gamma': 'scale', 'kernel': 'linear'}, 'svm_512_bert_768_tail': {'C': 0.01, 'gamma': 'scale', 'kernel': 'linear'}, 'svm_512_fasttext_300_tail': {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}, 'svm_512_glove_300_split': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_512_glove_50_split': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}, 'svm_512_glove_50_tail': {'C': 10, 'gamma': 'scale', 'kernel': 'linear'}, 'svm_512_fasttext_300_split': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_512_bert_768_split': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}, 'svm_512_glove_50_head': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'gaussian_256_bert_768_tail': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_fasttext_300_head': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_bert_768_head': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_fasttext_300_tail': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_bert_768_split': {'kernel': RBF(length_scale=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_fasttext_300_split': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_glove_50_head': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_glove_300_head': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_glove_50_split': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_glove_300_split': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_glove_300_tail': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_glove_50_tail': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'xgboost_512_glove_50_head': {'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_512_fasttext_300_tail': {'learning_rate': 0.1, 'n_estimators': 250}, 'xgboost_512_glove_300_head': {'learning_rate': 0.1, 'n_estimators': 250}, 'xgboost_512_glove_300_tail': {'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_512_glove_50_split': {'learning_rate': 0.1, 'n_estimators': 250}, 'xgboost_512_glove_300_split': {'learning_rate': 0.1, 'n_estimators': 250}, 'xgboost_512_glove_50_tail': {'learning_rate': 0.1, 'n_estimators': 200}, 'xgboost_512_fasttext_300_head': {'learning_rate': 0.1, 'n_estimators': 250}, 'xgboost_512_fasttext_300_split': {'learning_rate': 0.1, 'n_estimators': 150}, 'xgboost_512_bert_768_tail': {'learning_rate': 0.1, 'n_estimators': 250}, 'xgboost_512_bert_768_head': {'learning_rate': 0.1, 'n_estimators': 200}, 'xgboost_512_bert_768_split': {'learning_rate': 0.01, 'n_estimators': 50}, 'knn_256_bert_768_split': {'metric': 'euclidean', 'n_neighbors': 1}, 'knn_256_glove_50_tail': {'metric': 'euclidean', 'n_neighbors': 1}, 'knn_256_glove_50_head': {'metric': 'euclidean', 'n_neighbors': 1}, 'knn_256_bert_768_head': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_256_glove_50_split': {'metric': 'euclidean', 'n_neighbors': 1}, 'knn_256_glove_300_tail': {'metric': 'euclidean', 'n_neighbors': 1}, 'knn_256_fasttext_300_tail': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_256_fasttext_300_head': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_256_glove_300_head': {'metric': 'euclidean', 'n_neighbors': 1}, 'knn_256_glove_300_split': {'metric': 'euclidean', 'n_neighbors': 1}, 'knn_256_fasttext_300_split': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_256_bert_768_tail': {'metric': 'manhattan', 'n_neighbors': 1}, 'nb_512_glove_50_tail': {}, 'nb_512_glove_300_head': {}, 'nb_512_fasttext_300_head': {}, 'nb_512_glove_300_tail': {}, 'nb_512_glove_50_head': {}, 'nb_512_fasttext_300_tail': {}, 'nb_512_glove_50_split': {}, 'nb_512_glove_300_split': {}, 'nb_512_bert_768_head': {}, 'nb_512_fasttext_300_split': {}, 'nb_512_bert_768_tail': {}, 'nb_512_bert_768_split': {}, 'gaussian_512_glove_50_split': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_glove_300_split': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_glove_300_tail': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_glove_50_tail': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_glove_50_head': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_glove_300_head': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_bert_768_split': {'kernel': RBF(length_scale=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_bert_768_head': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_fasttext_300_tail': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_fasttext_300_split': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_bert_768_tail': {'kernel': RBF(length_scale=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_fasttext_300_head': {'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1)}, 'xgboost_256_bert_768_head': {'learning_rate': 0.1, 'n_estimators': 250}, 'xgboost_256_bert_768_split': {'learning_rate': 0.1, 'n_estimators': 250}, 'xgboost_256_fasttext_300_split': {'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_256_bert_768_tail': {'learning_rate': 0.1, 'n_estimators': 250}, 'xgboost_256_glove_300_tail': {'learning_rate': 0.1, 'n_estimators': 250}, 'xgboost_256_fasttext_300_head': {'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_256_glove_50_split': {'gamma': 0, 'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_256_glove_300_split': {'learning_rate': 0.1, 'n_estimators': 150}, 'xgboost_256_glove_50_tail': {'gamma': 0, 'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_256_fasttext_300_tail': {'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_256_glove_50_head': {'gamma': 0, 'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_256_glove_300_head': {'learning_rate': 0.1, 'n_estimators': 300}, 'svm_256_glove_50_head': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_256_bert_768_split': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_256_glove_50_tail': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_256_fasttext_300_split': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_256_fasttext_300_tail': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_256_glove_300_tail': {'C': 10, 'gamma': 'scale', 'kernel': 'linear'}, 'svm_256_bert_768_tail': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_256_glove_300_split': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_256_glove_50_split': {'C': 100, 'gamma': 'scale', 'kernel': 'linear'}, 'svm_256_fasttext_300_head': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_256_glove_300_head': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'svm_256_bert_768_head': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'}, 'nb_256_2007': {}, 'nb_256_2015': {}, 'nb_256_2001': {}, 'nb_256_2022': {}, 'knn_512_2015': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_512_2001': {'metric': 'euclidean', 'n_neighbors': 1}, 'knn_512_2007': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_512_2022': {'metric': 'manhattan', 'n_neighbors': 1}, 'svm_512_2007': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}, 'svm_512_2015': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}, 'svm_512_2001': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}, 'svm_512_2022': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}, 'gaussian_256_2007': {'kernel': RBF(length_scale=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_2015': {'kernel': RBF(length_scale=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_2001': {'kernel': RBF(length_scale=1) + WhiteKernel(noise_level=1)}, 'gaussian_256_2022': {'kernel': RBF(length_scale=1) + WhiteKernel(noise_level=1)}, 'xgboost_512_2015': {'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_512_2001': {'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_512_2007': {'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_512_2022': {'learning_rate': 0.1, 'n_estimators': 300}, 'knn_256_2015': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_256_2001': {'metric': 'euclidean', 'n_neighbors': 1}, 'knn_256_2007': {'metric': 'manhattan', 'n_neighbors': 1}, 'knn_256_2022': {'metric': 'manhattan', 'n_neighbors': 1}, 'nb_512_2007': {}, 'nb_512_2015': {}, 'nb_512_2001': {}, 'nb_512_2022': {}, 'gaussian_512_2007': {'kernel': RBF(length_scale=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_2015': {'kernel': RBF(length_scale=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_2001': {'kernel': RBF(length_scale=1) + WhiteKernel(noise_level=1)}, 'gaussian_512_2022': {'kernel': RBF(length_scale=1) + WhiteKernel(noise_level=1)}, 'xgboost_256_2015': {'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_256_2001': {'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_256_2007': {'learning_rate': 0.1, 'n_estimators': 300}, 'xgboost_256_2022': {'learning_rate': 0.1, 'n_estimators': 300}, 'svm_256_2007': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}, 'svm_256_2015': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}, 'svm_256_2001': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}, 'svm_256_2022': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}}

grid_search_params = {
    "nb" : (GaussianNB(), {}),
    "knn": (KNeighborsClassifier(), {
        'n_neighbors': range(1, 21),
        'metric': ['euclidean', 'manhattan', 'minkowski']}),
    "svm" : (SVC(), {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']}),
    "xgboost": (XGBClassifier(eval_metric='logloss'), {
        'n_estimators': range(50, 251, 50),
        'learning_rate': [0.01, 0.05, 0.1],
        }),
    "gaussian": (GaussianProcessClassifier(), {
        'kernel': [DotProduct() + WhiteKernel(), RBF(length_scale=1.0) + WhiteKernel()],
        })
}


def _get_model(model = "xgboost", param_string=None):
    """Helper function for extracting model"""
    if model == "svm":
        return SVC(probability=True, **training_params_from_gridsearch[param_string]) if param_string else SVC(probability=True)
    if model == "sgd":
        return SGDClassifier(**training_params_from_gridsearch[param_string]) if param_string else SGDClassifier()
    elif model == "nb":
        return GaussianNB(**training_params_from_gridsearch[param_string]) if param_string else GaussianNB()
    elif model == "knn":
        return KNeighborsClassifier(**training_params_from_gridsearch[param_string]) if param_string else KNeighborsClassifier()
    elif model == "xgboost":
        return XGBClassifier(eval_metric="logloss", **training_params_from_gridsearch[param_string]) if param_string else XGBClassifier(eval_metric="logloss")
    elif model == "gaussian":
        return GaussianProcessClassifier(**training_params_from_gridsearch[param_string]) if param_string else GaussianProcessClassifier()
    else:
        raise NotImplementedError()

def _save_model(model, saved_model_dir, name = 'sklearn_model.sav'):
    """Helper function for saving model to disk"""
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(saved_model_dir / name)
    pickle.dump(model, open(model_path, 'wb'))
    
    return model_path

# Training
def training(saved_model_dir, path, model_type, batch_size = None, grid_search_metric = "f2", random_search = False, test_path=None):
    """Method for training sklearn model.

    Args:
        saved_model_dir (str): Directory for saved model.
        path (str): Data path.
        model_type (str): Model type.
        batch_size (int, optional): If set, fit in iterations over data samples. Defaults to None.
        grid_search_metric (str, optional): Metric to use for grid search optimization. Defaults to "f2".
        random_search (bool, optional): If set, use random search instead of grid search. Defaults to False.
        test_path (str, optional): If set, run test on dataset at path. Defaults to None.
    """
    
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
        scoring = {'recall': make_scorer(recall_score), 'f1': make_scorer(f1_score), 'f2': make_scorer(fbeta_score, beta=2), 'precision': make_scorer(precision_score)}

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
        print("Best Recall score:", grid_search.cv_results_['mean_test_recall'][best_index])
        print("Best Precision score:", grid_search.cv_results_['mean_test_precision'][best_index])
        print("Best F1 score:", grid_search.cv_results_['mean_test_f1'][best_index])
        print("Best F2 score:", grid_search.cv_results_['mean_test_f2'][best_index])
        print("-" * 80)
        print("Best params:", grid_search.best_params_)
        print("Best estimator:", grid_search.best_estimator_)
        print("=" * 80)

    # If regular training
    else:
        grid_search_key = f"{model_type}_{path.split('.')[0].split('_')[-1]}_{'_'.join(path.split('.')[0].split('_')[-4:-1])}" if "embeddings" in str(path) else f"{model_type}_{path.split('.')[0].split('_')[4]}_{path.split('/')[-2]}"
        print(grid_search_key)

        # Get model
        model = _get_model(model_type, grid_search_key)

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
            print(X.shape)
            print(f"Created X in {round(time.time() - start_time, 0)} seconds. Size: {round(X.nbytes / 10**9, 2)}GB")
            y = np.array(labels)

            # Fit model
            print("Fitting model to data ...")
            start_time = time.time()
            model.fit(X, y)
            print(f"Fitted model to data in {round(time.time() - start_time, 0)} seconds")

        if test_path:
            data = read_h5(test_path)
            X = np.array([element.ravel().tolist() for element in data["emb_tensor"]])
            y_true = np.array(data["label"])
            y_pred = model.predict(X)
            metrics = get_metrics(y_pred, y_true)
            
            print("-"*50)
            print(f"TP: {metrics['tp']} | TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
            print(f"Accuracy: {round(metrics['accuracy'], 5)}")
            print(f"Precision: {round(metrics['precision'], 5)}")
            print(f"Recall: {round(metrics['recall'], 5)}")
            print(f"F1-score: {round(metrics['f1_score'], 5)}")
            print(f"F2-score: {round(metrics['f2_score'], 5)}")
            print("-"*50)
            
        # Save model
        _save_model(model, saved_model_dir=saved_model_dir)
        

# Training based on selected feature
SUPPORTED_EMBEDDINGS = ["glove", "glove_50", "fasttext", "bert"]
def train_embeddings(path: str, emb:str, model:str, batch_size = None, grid_search_metric = None, test_path=None):
    assert emb in SUPPORTED_EMBEDDINGS, "Embedding not supported!"
    training(saved_model_dir=Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / model / 'embeddings' / emb / Path(path).stem, path=path, model_type=model, batch_size=batch_size, grid_search_metric=grid_search_metric, test_path=test_path)

SUPPORTED_LIWC_DICTS = ["2022", "2015", "2007", "2001"]
def train_liwc(path: str, liwc_dict:str, model:str, batch_size = None, grid_search_metric = None, test_path=None):
    assert liwc_dict in SUPPORTED_LIWC_DICTS, "Liwc dictionary not supported!"
    training(saved_model_dir=Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / model / 'liwc' / liwc_dict / Path(path).stem, path=path, model_type=model, batch_size=batch_size, grid_search_metric=grid_search_metric, test_path=test_path)


click.option = partial(click.option, show_default=True)
@click.command()
@click.argument("path", nargs=1)
@click.option("-m", "--model", type=click.Choice(["svm", "sgd", "nb", "xgboost", "knn", "gaussian"]), default="svm", help="Model to select from saved models")
@click.option("-g", "--grid-search-metric", type=click.Choice(["f1", "recall", "recall_f1", "precision", None]), default=None, help="Perform grid search with given metric")
@click.option("-t", "--test-path", default=None, help="Test path if wanted to test after training")
def main(path, model, grid_search_metric, test_path):

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
        train_embeddings(path, emb_type, model, grid_search_metric = grid_search_metric, test_path=test_path)
        

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
        train_liwc(path, liwc_dict, model, grid_search_metric = grid_search_metric, test_path=test_path)

if __name__ == "__main__":
    main()