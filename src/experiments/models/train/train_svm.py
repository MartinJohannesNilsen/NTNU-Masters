from functools import partial
import os
from pathlib import Path
import click
import numpy as np
from sklearn.svm import SVC, SVR
import pickle   
from sklearn.model_selection import KFold
from tabulate import tabulate
import sys
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, get_average_metrics
from experiments.features.load_and_store_emb_batches import read_h5


# Training utitilites
def _get_svm_model():
    return SVR()

def _save_model(model, saved_model_dir, name = 'sklearn_model.sav'):
    # Save the model to disk
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(saved_model_dir / name)
    pickle.dump(model, open(model_path, 'wb'))
    
    return model_path
        

# Training
def training(saved_model_dir, path, batch_size = None, cross_validation_splits: int = None):

    # Get model
    model = _get_svm_model()
    
    # Run cross_val if number of splits is defined
    if cross_validation_splits:
        assert cross_validation_splits > 1, "n_splits needs to be more than 1!"

        # Read data
        data = read_h5(path)

        # Train data inputs X and labels y
        X = np.array([element.ravel().tolist() for element in data["emb_tensor"]]) # Flatten (512, emb_dim) into (512*emb_dim) with ravel, make list and output a numpy array
        y = np.array(data["label"])

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=cross_validation_splits, shuffle=True)

        # K-fold Cross Validation model evaluation
        fold_no = 1
        metrics = {}
        for train, test in kfold.split(X, y):    
            
            # Fit the model to data
            model = _get_svm_model()
            model.fit(X[train], y[train])
            
            # Get preds
            y_pred = model.predict(X[test])
            threshold = 0.5
            preds = [1 if pred > threshold else 0 for pred in y_pred]
            metrics[f"Fold {fold_no}"] = get_metrics(preds, y[test])

            fold_no += 1

        # Average
        metrics["Average"] = get_average_metrics(metrics_array=metrics.values())

        # 2D array for tabulate
        all = []
        for k, v in metrics.items():
            out = [k]
            for metric in v.values():
                out.append(round(metric, 3)) if metric else out.append(None)
            all.append(out)

        print(tabulate(all, headers=["Fold", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC"]))

    # If regular training
    else:
        if batch_size:
            n_samples_total = len(read_h5(path, col_name="idx"))
            # Iterate over all batches
            for i in range(0, n_samples_total, batch_size):
                
                # Read data
                data = read_h5(path, start=i, chunk_size=batch_size)

                # Train data inputs X and labels y
                X = np.array([element.ravel().tolist() for element in data["emb_tensor"]]) # Flatten (512, emb_dim) into (512*emb_dim) with ravel, make list and output a numpy array
                y = np.array(data["label"])

                # Fit model
                model.fit(X, y)
        
        else:
            # Read data
            data = read_h5(path)

            # Train data inputs X and labels y
            X = np.array([element.ravel().tolist() for element in data["emb_tensor"]]) # Flatten (512, emb_dim) into (512*emb_dim) with ravel, make list and output a numpy array
            y = np.array(data["label"])

            # Fit model
            model.fit(X, y)
            
        # Save model
        _save_model(model, saved_model_dir=saved_model_dir)
        

# Helper functions for feature based training
SUPPORTED_EMBEDDINGS = ["glove", "glove_50", "fasttext", "bert"]
def _embedding(path, emb_type = "glove", batch_size = None, cross_validation_splits: int = None):
    assert emb_type in SUPPORTED_EMBEDDINGS, "Embedding type not supported!"
    training(saved_model_dir=Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'svm' / 'embeddings' / emb_type / Path(path).stem, path=path, batch_size=batch_size, cross_validation_splits=cross_validation_splits)

SUPPORTED_LIWC_DICTS = ["2022", "2015", "2007", "2001"]
def _liwc(path, liwc_dict = "2022", batch_size = None, cross_validation_splits: int = None):
    assert liwc_dict in SUPPORTED_LIWC_DICTS, "LIWC dictionary version not supported!"
    training(saved_model_dir=Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'svm' / 'liwc' / f'{liwc_dict}' / Path(path).stem, path=path, batch_size=batch_size, cross_validation_splits=cross_validation_splits)


# Training based on selected feature
def train_embeddings(path: str, emb:str = "glove", cross_val_splits = None):
    assert emb in SUPPORTED_EMBEDDINGS, "Embedding not supported!"
    if cross_val_splits:
        _embedding(emb_type=emb, path=path, cross_validation_splits=cross_val_splits)
    else:
        _embedding(emb_type=emb, path=path, batch_size=32)


def train_liwc(path: str, liwc_dict:str = "2022", cross_val_splits = None):
    assert liwc_dict in SUPPORTED_LIWC_DICTS, "Liwc dictionary not supported!"
    if cross_val_splits:
        _liwc(path=path, liwc_dict=liwc_dict, cross_validation_splits=cross_val_splits)
    else:
        _liwc(path=path, liwc_dict=liwc_dict)


click.option = partial(click.option, show_default=True)
@click.command()
@click.argument("path", nargs=1)
def main(path):

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
        train_embeddings(path, emb_type)
        
    elif "LIWC" in path:
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
        train_liwc(path, liwc_dict)

if __name__ == "__main__":
    main()