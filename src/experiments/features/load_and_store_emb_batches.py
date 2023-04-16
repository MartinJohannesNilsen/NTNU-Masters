import pickle
import pandas as pd
from pathlib import Path
from csv import QUOTE_NONE
import os
import sys
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
from experiments.utils.word_embeddings import get_glove_word_vectors, get_fasttext_word_vectors, get_bert_word_embeddings


def replace_text_with_embedding(df: pd.DataFrame, emb_type = "glove"):
    assert emb_type == "glove" or emb_type == "fasttext" or emb_type == "bert", "emb_type not supported!"
    if emb_type == "glove":
        df["text"] = df["text"].map(lambda a: get_glove_word_vectors(a, sentence_length=512))
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

    store = pd.HDFStore(fpath) #bz2.BZ2File(fpath, "ab") # Compressed file stream bz2 format, ab for append byte format

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


def fetch_embeddings_from_hdf5(fpath: str, chunksize: int, column_names=None, get_df: bool = True, iterator: bool = False):
    """
    Function to fetch embeddings from file with given step size. Helps alleviate memory constraints with large embeddings sizes

    f: File object to fetch data from
    column_names: Name of columns of dataframe to be created
    n_rows: Amount of steps to fetch
    """

    if iterator:
        return iter(pd.read_hdf(fpath, iterator=True, chunksize=chunksize)) # pd_read_hdf with iterator=True returns a TableIterator. Need to call __iter__() to get pure iterator obj
    
    return pd.read_hdf(fpath, start=0, stop=chunksize, mode="r") if chunksize else pd.read_hdf(fpath, mode="r")


if __name__ == "__main__":
    
    data_folder = Path(os.path.abspath(__file__)).parents[2] / "dataset_creation" / "data" / "train_test"
    out_path = Path(os.path.abspath(__file__)).parents[1] / "features" / "embeddings"

    train_df = pd.read_csv(data_folder / "train_no_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    test_df = pd.read_csv(data_folder / "test_no_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    hold_out_df = pd.read_csv(data_folder / "shooter_hold_out_test.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    
    embeddings = ["glove", "fasttext", "bert"]
    
    for emb_type in embeddings:
        print(f"Type: {emb_type}, train")
        embedding_train_df = train_df.copy()
        create_and_store_embeddings(embedding_train_df, out_path / f"train_sliced_stair_twitter_{emb_type}.h5", emb_type, 500)        

        print(f"Type: {emb_type}, test")
        embedding_test_df = test_df.copy()
        create_and_store_embeddings(embedding_test_df, out_path / f"test_sliced_stair_twitter_{emb_type}.h5", emb_type, 500)

        print(f"Type: {emb_type}, hold out")
        embedding_hold_out_df = hold_out_df.copy()
        create_and_store_embeddings(embedding_test_df, out_path / f"hold_out_test_sliced_stair_twitter_{emb_type}.h5", emb_type, 500)


"""     # Decompress and load data into df
    data_iterator = iter(fetch_embeddings_from_hdf5(to_path, chunksize=10, iterator=True))
    items = next(data_iterator)
    try:
        i = 1
        while True:
            print(f"Chunk {i}\n", items)
            items = next(data_iterator)
            i += 1
    except StopIteration:
        pass """