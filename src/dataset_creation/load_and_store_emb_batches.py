import pickle
import pandas as pd
from pathlib import Path
from csv import QUOTE_NONE
import os
import sys
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from experiments.utils.word_embeddings import get_glove_word_vectors, get_fasttext_word_vectors, get_bert_word_embeddings, _tokenize_with_preprocessing
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

    """ store = pd.HDFStore(fpath) #bz2.BZ2File(fpath, "ab") # Compressed file stream bz2 format, ab for append byte format

    def embed_and_store_rows(rows):
        rows = replace_text_with_embedding(rows, emb_type=emb_type)
        rows["text"] = rows["text"].map(lambda a: a.numpy())
        #print(len(str(rows.values[0][1])))
        rows["text"] = rows["text"].apply(str)
        #print(rows["text"])
        store.append("row_frame", rows, format="t", data_columns=df.columns, min_itemsize={"text": 500, "name": 24})

    pos = 0
    while pos + step_size < len(df.index):
        rows = df.iloc[pos:pos+step_size]
        embed_and_store_rows(rows)
        rows = [] # Hacky way to free up memory :)
        pos += step_size
    
    if pos < len(df.index):
        rows = df.iloc[pos:]
        embed_and_store_rows(rows)
        rows = []
    
    store.close() """

    store = h5py.File(fpath, "a")

    def embed_rows_as_numpy(rows):
        rows = replace_text_with_embedding(rows, emb_type=emb_type)
        rows["text"] = rows["text"].map(lambda a: a.numpy())

        rows_e = []
        for row in rows["text"].values:
            temp = []
            for word in row:
                temp.append(word.tolist())
        
            rows_e.append(row)

        out_cols = {
            "date": np.array(rows["date"].values, dtype=h5py.special_dtype(vlen=str)),
            "embeddings": np.array(rows_e),
            "name": np.array(rows["name"].values, dtype=h5py.special_dtype(vlen=str)),
            "label": np.array(rows["label"].values, dtype=int),
        }

        return out_cols
    

    def first_time_setup_dataset(data, sentence_len, emb_dim):
        """
        First time dataset is accessed, it has to be created. Quick setup with dim0 = None to allow for resizing later
        """

        store.create_dataset("date", compression="gzip", data=data["date"], chunks=True, maxshape=data["date"].shape)

        emb_data_shape = (len(data["embeddings"]), data["embeddings"][0].shape[0], data["embeddings"][0].shape[1])

        print(emb_data_shape)
        print(data["embeddings"].shape)

        store.create_dataset("emb_tensor", compression="gzip", data=data["embeddings"], chunks=True, maxshape=emb_data_shape, dtype=np.float32)
        store.create_dataset("name", compression="gzip", data=data["name"], chunks=True, maxshape=data["name"].shape)
        store.create_dataset("label", compression="gzip", data=data["label"], chunks=True, maxshape=data["label"].shape)


    def resize_and_append_datasets(data, chunk_size):
        """
        rows: Data to be stored
        chunk_size: Increment to increase size of dataset with
        Resize dim0 of dataset to fit new data
        """

        store["date"].resize(store["date"].shape[0] + chunk_size, axis=0)
        store["date"][-chunk_size:] = data["date"]

        store["emb_tensor"].resize(store["emb_tensor"].shape[0] + chunk_size, axis=0)
        store["emb_tensor"][-chunk_size:] = data["embeddings"]

        store["name"].resize(store["name"].shape[0] + chunk_size, axis=0)
        store["name"][-chunk_size:] = data["name"]

        store["label"].resize(store["label"].shape[0] + chunk_size, axis=0)
        store["label"][-chunk_size:] = data["label"]


    sentence_len = -1
    emb_dim = 300

    # Setup data storage

    pos = 0
    while pos + step_size < len(df.index):
        rows = df.iloc[pos:pos+step_size]
        data = embed_rows_as_numpy(rows)
        #print(data)
        
        if pos == 0:
            sentence_len = data["embeddings"][0].shape[0]
            first_time_setup_dataset(data, sentence_len, emb_dim)

        else:   
            resize_and_append_datasets(data, step_size)
        
        rows = []
        pos += step_size

    if pos < len(df.index):
        rows = df.iloc[pos:]
        data = embed_rows_as_numpy(rows)

        if pos == 0:
            #print(rows[0][1].shape[0])
            sentence_len = data["embeddings"][0].shape[0]
            first_time_setup_dataset(data, sentence_len, emb_dim)

        else:
            resize_and_append_datasets(data, pos-len(df.index))
        
        rows = [] # Hacky way to free up memory :)
    

    print(f"I am on pos {pos} and 'data' chunk has shape: {store['emb_tensor'].shape}")

    store.close()


def fetch_embeddings_from_file(f, column_names=None, n_steps=500, get_df: bool = False):
    """
    Function to fetch embeddings from file with given step size. Helps alleviate memory constraints with large embeddings sizes

    f: File object to fetch data from
    column_names: Name of columns of dataframe to be created
    n_rows: Amount of steps to fetch
    """
    
    data = [] #dict.fromkeys(column_names, []) #pd.DataFrame(columns=column_names)
    try:
        for _ in range(n_steps):
            row = pickle.load(f)
            print(row)
            data.append(row)
            #print(rows_fetched)
            """ for row in rows_fetched:
                data.append(row) """
        
    except EOFError:
        pass
    
    if get_df:
        fetched_df = None
        if len(data) > 0:
            fetched_df = pd.DataFrame(data)
        else:
            print(f"COULD NOT FETCH DATA FROM PICKLED FILE. FILE IS EITHER EMPTY OR FETCH FAILED")
        return fetched_df

    #print(fetched_df.iloc[0][1])
    return data


def fetch_rows_from_h5(fpath: str, start, chunk_size):
    """
    Function to fetch embeddings from file with given step size. Helps alleviate memory constraints with large embeddings sizes

    f: File object to fetch data from
    column_names: Name of columns of dataframe to be created
    n_rows: Amount of steps to fetch
    """

    fetched_data = None

    with h5py.File(fpath, "r") as f:
        fetched_data = [
            f["date"][start:start+chunk_size],
            f["emb_tensor"][start:start+chunk_size],
            f["name"][start:start+chunk_size],
            f["label"][start:start+chunk_size]
        ]


    return fetched_data


if __name__ == "__main__":
    
    data_folder = Path(os.path.abspath(__file__)).parents[0] / "data" / "train_test"
    out_path = Path(os.path.abspath(__file__)).parents[1] / "experiments" / "features" / "embeddings"

    train_df = pd.read_csv(data_folder / "train_no_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    test_df = pd.read_csv(data_folder / "test_no_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    hold_out_df = pd.read_csv(data_folder / "shooter_hold_out_test.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    
    embeddings = ["glove", "fasttext", "bert"]
    
    """ for emb_type in embeddings:
        print(f"Type: {emb_type}, train")
        embedding_train_df = train_df.copy()
        create_and_store_embeddings(embedding_train_df, out_path / f"train_sliced_stair_twitter_{emb_type}.h5", emb_type, 500)        

        print(f"Type: {emb_type}, test")
        embedding_test_df = test_df.copy()
        create_and_store_embeddings(embedding_test_df, out_path / f"test_sliced_stair_twitter_{emb_type}.h5", emb_type, 500)

        print(f"Type: {emb_type}, hold out")
        embedding_hold_out_df = hold_out_df.copy()
        create_and_store_embeddings(embedding_hold_out_df, out_path / f"hold_out_test_sliced_stair_twitter_{emb_type}.h5", emb_type, 500) """
    
    for emb_type in embeddings:
        print(f"Type: {emb_type}, hold_out")
        embedding_hold_out_df = hold_out_df.copy()
        create_and_store_embeddings(embedding_hold_out_df, out_path / f"hold_out_test_sliced_stair_twitter_{emb_type}.h5", emb_type, 500)

    for emb_type in embeddings:
        fetched_data = fetch_rows_from_h5(out_path / f"hold_out_test_sliced_stair_twitter_{emb_type}.h5", start=0, chunk_size=5)
        print(fetched_data[1].shape)