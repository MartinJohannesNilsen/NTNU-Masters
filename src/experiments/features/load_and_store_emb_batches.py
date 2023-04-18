from typing import Union
import pandas as pd
from pathlib import Path
from csv import QUOTE_NONE
import os
import sys
import torch
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from utils.word_embeddings import get_glove_word_vectors, get_fasttext_word_vectors, get_bert_word_embeddings
import h5py
import numpy as np


def replace_text_with_embedding(df: pd.DataFrame, emb_type = "glove"):
    assert emb_type == "glove" or emb_type == "fasttext" or emb_type == "bert", "emb_type not supported!"
    if emb_type == "glove":
        df["text"] = df["text"].map(lambda a: get_glove_word_vectors(a, sentence_length=512, size_small=False))
    elif emb_type == "fasttext":
        df["text"] = df["text"].map(lambda a: get_fasttext_word_vectors(a, sentence_length=512))
    elif emb_type == "bert":
        df["text"] = df["text"].map(lambda a: get_bert_word_embeddings(a, sentence_length=512))
    else:
        return df
    
    df = df[df['text'].notna()]

    return df

def create_and_store_embeddings(df: pd.DataFrame, fpath: str, emb_type: str, step_size: int = 200):
    """
    Function to create and store embeddings to file with given step size. Helps alleviate memory constraints with large embeddings sizes

    df: Dataframe containing text from shooters
    fpath: path to file, file format should be h5 (hdf5)
    step_size: Amount of rows to be processed at once
    """

    store = h5py.File(fpath, "a")

    def replace_empty(string):
        return " " if not isinstance(string, str) else string

    def embed_rows_as_numpy(rows):
        rows["date"] = rows["date"].map(lambda a: replace_empty(a)) # Avoid conflict with h5py. None is treated as object type. Convert None to " "

        rows = replace_text_with_embedding(rows, emb_type=emb_type)
        rows["text"] = rows["text"].map(lambda a: a.numpy())

        # Converting to numpy compatible arrays so we can convert to multidim np arrays for storage
        rows_e = []
        for row in rows["text"].values:
            temp = []
            for word in row:
                temp.append(word.tolist())
        
            rows_e.append(row)

        out_cols = {
            "idx": rows.index,
            "date": np.array(rows["date"].values, dtype=h5py.special_dtype(vlen=str)),
            "embeddings": np.array(rows_e),
            "name": np.array(rows["name"].values, dtype=h5py.special_dtype(vlen=str)),
            "label": np.array(rows["label"].values, dtype=int),
        }
        #print(out_cols["idx"], out_cols["name"], out_cols["label"])

        return out_cols
    

    def first_time_setup_dataset(data):
        """
        First time dataset is accessed, it has to be created. Quick setup with dim0 = None to allow for resizing later
        """

        store.create_dataset("idx", compression="gzip", data=data["idx"], chunks=True, maxshape=(None, ))

        print("date: ", data["date"])
        store.create_dataset("date", compression="gzip", data=data["date"], chunks=True, maxshape=(None, ))

        emb_data_shape = (None, data["embeddings"][0].shape[0], data["embeddings"][0].shape[1])

        print(emb_data_shape)
        print(data["embeddings"].shape)

        store.create_dataset("emb_tensor", compression="gzip", data=data["embeddings"], chunks=True, maxshape=emb_data_shape, dtype=np.float32)
        store.create_dataset("name", compression="gzip", data=data["name"], chunks=True, maxshape=(None, ))
        store.create_dataset("label", compression="gzip", data=data["label"], chunks=True, maxshape=(None, ))


    def resize_and_append_datasets(data):
        """
        rows: Data to be stored
        chunk_size: Increment to increase size of dataset with
        Resize dim0 of dataset to fit new data
        """
        print(f"Current ds idx size: {store['idx'].shape[0]}\nCurrent rows shape: {data['idx'].shape[0]}")
        store["idx"].resize(store["idx"].shape[0] + data["idx"].shape[0], axis=0)
        print(f"Current ds idx size: {store['idx'].shape[0]}\nCurrent rows shape: {data['idx'].shape[0]}")
        store["idx"][-data["idx"].shape[0]:] = data["idx"]

        store["date"].resize(store["date"].shape[0] + data["date"].shape[0], axis=0)
        store["date"][-data["date"].shape[0]:] = data["date"]

        store["emb_tensor"].resize(store["emb_tensor"].shape[0] + data["embeddings"].shape[0], axis=0)
        store["emb_tensor"][-data["embeddings"].shape[0]:] = data["embeddings"]

        store["name"].resize(store["name"].shape[0] + data["name"].shape[0], axis=0)
        store["name"][-data["name"].shape[0]:] = data["name"]

        store["label"].resize(store["label"].shape[0] + data["label"].shape[0], axis=0)
        store["label"][-data["label"].shape[0]:] = data["label"]

    # Setup data storage

    pos = 0
    while pos + step_size < len(df.index):
        rows = df.iloc[pos:pos+step_size]
        data = embed_rows_as_numpy(rows)
        
        if pos == 0:
            first_time_setup_dataset(data)

        else:   
            resize_and_append_datasets(data)
        
        print(f"I am on pos {pos} and emb ds has shape: {store['emb_tensor'].shape}")
        rows = []
        pos += step_size

    if pos < len(df.index):
        rows = df.iloc[pos:]
        data = embed_rows_as_numpy(rows)

        if pos == 0:
            first_time_setup_dataset(data)

        else:
            resize_and_append_datasets(data)
        
        rows = [] # Hacky way to free up memory :)
        print(f"I am on pos {len(df.index) - pos} and 'data' chunk has shape: {store['emb_tensor'].shape}")
    

    store.close()


def fetch_data_from_h5(fpath: str, col_name: str = None, start: int = None, chunk_size: int = None, tolist = False) -> Union[list, dict]:
    """Function to fetch data from h5py data store. For alleviating memory constraints with large embeddings sizes, parameters 'start' and 'chunk_size' can be utilized for fetching given range.

    Args:
        fpath (str): Path to file.
        col_name (str, optional): If defined, this is the only column which will be returned. Defaults to None.
        start (int, optional): The start index in the defined range of rows to return. Defaults to None.
        chunk_size (int, optional): The size of the chunk of rows to be returned. Defaults to None.
        tolist (bool, optional): Return list instead of dictionary. Defaults to False.

    Returns:
        Union[list, dict]: Either a dictionary or a list depending on tolist parameter. Defaults to dict.
    """

    fetched_data = None

    with h5py.File(fpath, "r") as f:
        if col_name:
            if col_name in ["idx", "date", "emb_tensor", "name", "label"]:
                fetched_data = f[col_name][start:start+chunk_size if (start != None and chunk_size != None) else None:None]
                if col_name in ["date", "name", "label"]:
                    fetched_data = fetched_data.astype(str)
                elif col_name == "emb_tensor":
                    fetched_data = [torch.from_numpy(tensor) for tensor in fetched_data]
                elif col_name == "idx":
                    fetched_data = fetched_data.astype(int)
            else:
                print("Non-existent column name!")
                sys.exit(1)
        else:
            fetched_data = {
                "idx": list(f["idx"][start:start+chunk_size if (start != None and chunk_size != None) else None:None].astype(int)),
                "date": list(f["date"][start:start+chunk_size if (start != None and chunk_size != None) else None:None].astype(str)),
                "emb_tensor": [torch.from_numpy(tensor) for tensor in (f["emb_tensor"][start:start+chunk_size] if (start != None and chunk_size != None) else f["emb_tensor"])],
                "name": list(f["name"][start:start+chunk_size if (start != None and chunk_size != None) else None:None].astype(str)),
                "label": list(f["label"][start:start+chunk_size if (start != None and chunk_size != None) else None:None].astype(str))
            }
            if tolist:
                fetched_data = list(fetched_data.values())

    return fetched_data


if __name__ == "__main__":
    
    data_folder = Path(os.path.abspath(__file__)).parents[2] / "dataset_creation" / "data" / "train_test"
    out_path = Path(os.path.abspath(__file__)).parents[1] / "features" / "embeddings"

    train_df = pd.read_csv(data_folder / "train_no_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")[0:602]
    test_df = pd.read_csv(data_folder / "test_no_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    hold_out_df = pd.read_csv(data_folder / "shooter_hold_out_test.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    
    embeddings = ["bert"]
    
    # for emb_type in embeddings:
    #     print(f"Type: {emb_type}, hold_out")
    #     embedding_hold_out_df = train_df.copy()
    #     create_and_store_embeddings(embedding_hold_out_df, out_path / f"hold_out_test_sliced_stair_twitter_{emb_type}.h5", emb_type, 200)

    for emb_type in embeddings:
        fetched_data = fetch_data_from_h5(out_path / f"hold_out_test_sliced_stair_twitter_{emb_type}.h5", start=0, chunk_size=5)
        print(fetched_data)