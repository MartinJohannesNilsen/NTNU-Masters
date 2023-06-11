from functools import partial
from typing import Union
import click
import pandas as pd
from pathlib import Path
from csv import QUOTE_NONE
import os
import sys
import torch
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
import h5py
import numpy as np
from word_emb_utils import get_emb_model, embed_and_pad


data_folder = Path(os.path.abspath(__file__)).parents[3] / "data" / "processed_data" / "train_test" / "csv"
out_path = Path(os.path.abspath(__file__)).parents[3] / "features" / "embeddings"

name_to_dim = {
    "glove_50": "",
    "glove": "_300",
    "fasttext": "_300",
    "bert": "_768"
}

#  DATA ACCESS

def get_dfs():
    dfs = {
        "shooter_hold_out_256": pd.read_csv(data_folder / "shooter_hold_out_256.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        "shooter_hold_out_512": pd.read_csv(data_folder / "shooter_hold_out_512.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        
        "train_sliced_stair_twitter_512": pd.read_csv(data_folder / "train_sliced_stair_twitter_512.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        "train_sliced_stair_twitter_256": pd.read_csv(data_folder / "train_sliced_stair_twitter_256.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        "test_sliced_stair_twitter_512": pd.read_csv(data_folder / "test_sliced_stair_twitter_512.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        "test_sliced_stair_twitter_256": pd.read_csv(data_folder / "test_sliced_stair_twitter_256.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        "train_no_stair_twitter_512": pd.read_csv(data_folder / "train_no_stair_twitter_512.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        "train_no_stair_twitter_256": pd.read_csv(data_folder / "train_no_stair_twitter_256.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        "test_no_stair_twitter_512": pd.read_csv(data_folder / "test_no_stair_twitter_512.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        "test_no_stair_twitter_256": pd.read_csv(data_folder / "test_no_stair_twitter_256.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        "val_sliced_stair_twitter_512": pd.read_csv(data_folder / "val_sliced_stair_twitter_512.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        "val_sliced_stair_twitter_256": pd.read_csv(data_folder / "val_sliced_stair_twitter_256.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        "val_no_stair_twitter_512": pd.read_csv(data_folder / "val_no_stair_twitter_512.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
        "val_no_stair_twitter_256": pd.read_csv(data_folder / "val_no_stair_twitter_256.csv", sep="‎", quoting=QUOTE_NONE, engine="python"),
    }

    return dfs


# READING H5PY

def read_h5(fpath: str, col_name: str = None, start: int = None, chunk_size: int = None, tolist = False, keep_tensor_as_ndarray=False) -> Union[list, dict]:
    """Function to fetch data from h5py data store. For alleviating memory constraints with large embeddings sizes, parameters 'start' and 'chunk_size' can be utilized for fetching given range.

    Args:
        fpath (str): Path to file.
        col_name (str, optional): If defined, this is the only column which will be returned. Defaults to None.
        start (int, optional): The start index in the defined range of rows to return. Defaults to None.
        chunk_size (int, optional): The size of the chunk of rows to be returned. Defaults to None.
        tolist (bool, optional): Return list instead of dictionary. Defaults to False.
        keep_tensor_as_ndarray (bool, optional): Return ndarray instead of tensor. Defaults to False.

    Returns:
        Union[list, dict]: Either a dictionary or a list depending on tolist parameter. Defaults to dict.
    """

    fetched_data = None

    with h5py.File(fpath, "r") as f:
        if col_name:
            if col_name in ["idx", "date", "emb_tensor", "name", "label"]:
                fetched_data = f[col_name][start:start+chunk_size if (start != None and chunk_size != None) else None:None]
                if col_name in ["date", "name"]:
                    fetched_data = fetched_data.astype(str)
                elif col_name in ["idx", "label"]:
                    fetched_data = fetched_data.astype(int)
                elif col_name == "emb_tensor":
                    if not keep_tensor_as_ndarray:
                        fetched_data = [torch.from_numpy(tensor) for tensor in fetched_data]
            else:
                print("Non-existent column name!")
                sys.exit(0)
        else:
            fetched_data = {
                "idx": list(f["idx"][start:start+chunk_size if (start != None and chunk_size != None) else None:None].astype(int)),
                "date": list(f["date"][start:start+chunk_size if (start != None and chunk_size != None) else None:None].astype(str)),
                "emb_tensor": f["emb_tensor"] if keep_tensor_as_ndarray else [torch.from_numpy(tensor) for tensor in (f["emb_tensor"][start:start+chunk_size] if (start != None and chunk_size != None) else f["emb_tensor"])],
                "name": list(f["name"][start:start+chunk_size if (start != None and chunk_size != None) else None:None].astype(str)),
                "label": list(f["label"][start:start+chunk_size if (start != None and chunk_size != None) else None:None].astype(int))
            }
            if tolist:
                fetched_data = list(fetched_data.values())

    return fetched_data


def create_and_store_embeddings(df: pd.DataFrame, fpath: str, emb_model, step_size: int = 200, max_len: int = 512, pad_pos: str = "tail", emb_type: str = "glove"):
    """
    Function to create and store embeddings to file with given step size. Helps alleviate memory constraints with large embeddings sizes

    df: Dataframe containing text from shooters
    fpath: path to file, file format should be h5 (hdf5)
    step_size: Amount of rows to be processed at once
    emb_dim: Dimension of word embeddings to be created
    """

    store = h5py.File(fpath, "a")

    def replace_empty(string: str):
        return " " if not isinstance(string, str) else string

    def embed_rows_as_numpy(rows: pd.DataFrame):
        """
        Function to embed all rows with given embedding model. 
        Converts the tensors to numpy arrays to comply with h5py storage standards.

        Args:
            rows (pd.DataFrame): Dataframe rows containing the columns [date,text,name,label]

        Returns:
            out_cols (dict): Dictionary containing all columns of the dataframe provided as input. Each column is now given as a np array
                             The text column has been embedded with the given word embedding scheme.
        """

        rows["date"] = rows["date"].map(lambda a: replace_empty(a)) # Avoid conflict with h5py. None is treated as object type. Convert None to " "
        rows["text"] = rows["text"].map(lambda text: embed_and_pad(text, emb_model=emb_model, max_len=max_len, pad_pos=pad_pos, emb_type=emb_type))
        rows = rows[rows['text'].notna()]

        res_rows = rows["text"].copy().values
        
        # Converting to numpy compatible arrays so we can convert to multidim np arrays for storage
        emb_rows = []
        lengths = []
        for row in res_rows:
            embs, length = row[0], row[1]
            emb_rows.append(embs.numpy())
            lengths.append(length)

        out_cols = {
            "idx": rows.index,
            "date": np.array(rows["date"].values, dtype=h5py.special_dtype(vlen=str)),
            "emb_tensor": np.array(emb_rows),
            "name": np.array(rows["name"].values, dtype=h5py.special_dtype(vlen=str)),
            "label": np.array(rows["label"].values, dtype=int),
            "length": np.array(lengths, dtype=int)
        }

        return out_cols


    def first_time_setup_dataset(data):
        """
        First time dataset is accessed, it has to be created. Quick setup with dim0 = None to allow for resizing later
        """

        store.create_dataset("idx", compression="gzip", data=data["idx"], chunks=True, maxshape=(None, ))

        print("date: ", data["date"])
        store.create_dataset("date", compression="gzip", data=data["date"], chunks=True, maxshape=(None, ))

        emb_data_shape = (None, data["emb_tensor"][0].shape[0], data["emb_tensor"][0].shape[1])

        print(f"emb_data_shape: {emb_data_shape}")
        print(f"Shape of first data sent to emb_data_column: {data['emb_tensor'].shape}")

        store.create_dataset("emb_tensor", compression="gzip", data=data["emb_tensor"], chunks=True, maxshape=emb_data_shape, dtype=np.float32)
        store.create_dataset("name", compression="gzip", data=data["name"], chunks=True, maxshape=(None, ))
        store.create_dataset("label", compression="gzip", data=data["label"], chunks=True, maxshape=(None, ))
        store.create_dataset("length", compression="gzip", data=data["length"], chunks=True, maxshape=(None, ))



    def resize_and_append_datasets(data):
        """
        rows: Data to be stored
        chunk_size: Increment to increase size of dataset with
        Resize dim0 of dataset to fit new data
        """

        cols = ["idx", "date", "emb_tensor", "name", "label", "length"]

        for colname in cols:
            store[colname].resize(store[colname].shape[0] + data[colname].shape[0], axis=0)
            store[colname][-data[colname].shape[0]:] = data[colname] # Insert data into the newly extended part of the dataset

    # Iterate through the dataframe step_size rows at a time
    pos = 0
    while pos + step_size < len(df.index):
        rows = df.iloc[pos:pos+step_size]
        data = embed_rows_as_numpy(rows)
        
        if pos == 0:
            # Setup data storage
            first_time_setup_dataset(data)

        else:   
            resize_and_append_datasets(data)
        
        rows = []
        pos += step_size

    # If there are n remaining rows, but step_size is too large, do one last iteration with n step_size to avoid overshooting
    if pos < len(df.index):
        rows = df.iloc[pos:]
        data = embed_rows_as_numpy(rows)

        if pos == 0:
            first_time_setup_dataset(data)

        else:
            resize_and_append_datasets(data)
        
        rows = [] # Hacky way to free up memory :)
    
    store.close()


def create_and_store_all_embs_of_type(dfs, emb_type: str, pad_pos = None, step_size: int = 200, max_len: int = 256):
    """
    Convenience function to perform all steps of creating embs and storing them

    dfs: List of dataframes to be processed
    emb_type: Embedding type to be used
    """
    assert emb_type in ["glove_50", "glove", "fasttext", "bert"], "Embedding type not supported!"

    emb_model = get_emb_model(emb_type)
    purpose = ["train", "test", "val"]
    stair_twitter = ["sliced_stair"]

    for p in purpose:
        for st in stair_twitter:
            print(f"Type: {emb_type}, {p}, {pad_pos}, {st}, {max_len}")
            df = dfs[f"{p}_{st}_twitter_{max_len}"].copy()
            create_and_store_embeddings(df, fpath=out_path / f"{p}_{st}_twitter_{emb_type}{name_to_dim[emb_type]}_{pad_pos}_{max_len}.h5", emb_model=emb_model, step_size=step_size, max_len=max_len, pad_pos=pad_pos, emb_type=emb_type)    
            df = None

def create_and_store_shooter_hold_out_embs(dfs, emb_type: str, pad_pos = None, step_size: int = 200, max_len: int = 256):
    """
    Convenience function to perform all steps of creating embs and storing them

    dfs: List of dataframes to be processed
    emb_type: Embedding type to be used
    """
    assert emb_type in ["glove_50", "glove", "fasttext", "bert"], "Embedding type not supported!"

    emb_model = get_emb_model(emb_type)
    df = dfs[f"shooter_hold_out_{max_len}"].copy()
    create_and_store_embeddings(df, fpath=out_path / f"shooter_hold_out_{emb_type}{name_to_dim[emb_type]}_{pad_pos}_{max_len}.h5", emb_model=emb_model, step_size=step_size, max_len=max_len, pad_pos=pad_pos, emb_type=emb_type)    


@click.command()
@click.option("-e", "--emb", type=click.Choice(["fasttext", "glove", "bert", "glove_50"]) , help="Embedding type to be used for training")
@click.option("-l", "--length", type=click.INT, help="Max length of sequence")
@click.option("-p", "--pad_pos", type=click.Choice(["head", "tail", "split"]), help="Position of padding")
def main(emb = None, length = None, pad_pos = None):
    dfs = get_dfs()
    create_and_store_all_embs_of_type(dfs, emb_type="glove_50", max_len=256, pad_pos="tail", step_size=200)



if __name__ == "__main__":
    main()
    
    # dfs = get_dfs()
    # create_and_store_all_embs_of_type(dfs, "bert", step_size=200)

    # with h5py.File(out_path / "train_no_stair_twitter_glove_50_head_256.h5", "r") as f:
    #     print("printing emb tensors for test...")
    #     for i in range(3):
    #         print(f["emb_tensor"][i])
    