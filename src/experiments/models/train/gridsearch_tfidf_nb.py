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

if __name__ == "__main__":
    train(256)