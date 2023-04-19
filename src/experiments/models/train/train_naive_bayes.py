import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import pickle   
from sklearn.model_selection import KFold
from tabulate import tabulate
import sys
experiments_dir = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.append(experiments_dir)
from experiments.utils.metrics import get_metrics, get_average_metrics
from experiments.features.load_and_store_emb_batches import read_h5


def train_embeddings(embedding_type = "bert", cross_validation_splits: int = None):
    assert embedding_type in ["glove", "fasttext", "bert"], "Embedding not supported!"

    model = GaussianNB()

    def save_model(model, name = 'sklearn_model.sav'):
        # Save the model to disk
        model_dir = Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'naive_bayes' / f'{embedding_type}_embeddings'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(model_dir / name)
        pickle.dump(model, open(model_path, 'wb'))
        
        return model_path
    
    # Get data from h5 store
    path = str(Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings" / f"train_sliced_stair_twitter_{embedding_type}.h5")
    # path = str(Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings" / f"hold_out_test_sliced_stair_twitter_{embedding_type}.h5")
    
    if cross_validation_splits:

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
            model = GaussianNB()
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

    else:
        BATCH_SIZE = 32
        n_samples_total = len(read_h5(path, col_name="idx"))
        
        # Iterate over all batches
        for i in range(0, n_samples_total, BATCH_SIZE):
            
            # Read data
            data = read_h5(path, start=i, chunk_size=BATCH_SIZE)

            # Train data inputs X and labels y
            X = np.array([element.ravel().tolist() for element in data["emb_tensor"]]) # Flatten (512, emb_dim) into (512*emb_dim) with ravel, make list and output a numpy array
            y = np.array(data["label"])

            # Fit model
            model.fit(X, y)
        
        # Save model
        save_model(model)
        

if __name__ == "__main__":
    embeddings = ["glove", "fasttext", "bert"]
    # embeddings = ["bert"]
    for emb_type in embeddings:
       print(emb_type)
       train_embeddings(embedding_type=emb_type)
    #    train_embeddings(embedding_type=emb_type, cross_validation_splits=2)

