import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import pickle    

def train_embeddings(embedding_type = "bert"):
    assert embedding_type in ["glove", "fasttext", "bert"], "Embedding not supported!"

    # Get pickled dataframe with embeddings
    data_path = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings" / f"DEMO_train_no_stair_twitter_{embedding_type}.pkl"
    df = pd.read_pickle(data_path, compression="bz2")

    # Train data inputs X and labels y
    X = np.array([element.ravel().tolist() for element in df["text"].values]) # Flatten (512, emb_dim) into (512*emb_dim) with ravel, make list and output a numpy array
    y = np.array(df["label"].values)

    # Fit the model to data
    model = SVC()
    model.fit(X, y)

    # Save the model to disk
    model_dir = Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'svm' / f'{embedding_type}_embeddings'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(model_dir / 'sklearn_model.sav')
    pickle.dump(model, open(model_path, 'wb'))
    return model_path, model

if __name__ == "__main__":
    for emb_type in ["glove", "fasttext", "bert"]:
        train_embeddings(embedding_type=emb_type)
    