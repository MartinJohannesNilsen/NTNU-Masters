import pandas as pd
from pathlib import Path
from csv import QUOTE_NONE
from load_and_store_emb_batches import create_and_store_embeddings
import os

if __name__ == "__main__":
    
    data_folder = Path(os.path.abspath(__file__)).parents[2] / "dataset_creation" / "data" / "train_test"
    out_path = Path(os.path.abspath(__file__)).parents[1] / "features" / "embeddings"

    train_df = pd.read_csv(data_folder / "train_sliced_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    test_df = pd.read_csv(data_folder / "test_sliced_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    hold_out_df = pd.read_csv(data_folder / "shooter_hold_out_test.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    
    
    print(f"Type: bert, train")
    embedding_train_df = train_df.copy()
    create_and_store_embeddings(embedding_train_df, out_path / f"train_sliced_stair_twitter_bert.h5", "bert", 200)        

    print(f"Type: bert, test")
    embedding_test_df = test_df.copy()
    create_and_store_embeddings(embedding_test_df, out_path / f"test_sliced_stair_twitter_bert.h5", "bert", 200)

    print(f"Type: bert, hold out")
    embedding_hold_out_df = hold_out_df.copy()
    create_and_store_embeddings(embedding_hold_out_df, out_path / f"hold_out_test_sliced_stair_twitter_bert.h5", "bert", 200)