import pickle
import pandas as pd
from pathlib import Path
from csv import QUOTE_NONE
import os
import sys
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from experiments.utils.word_embeddings import get_glove_word_vectors, get_fasttext_word_vectors, get_bert_word_embeddings
import bz2


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
    fpath: path to file, no need for extension as file is stored as bz2 file
    step_size: Amount of rows to be processed at once
    """

    ofile = bz2.BZ2File(fpath, "ab") # Compressed file stream bz2 format, ab for append byte format

    pos = 0
    while pos + step_size < len(df.index):
        rows = df.iloc[pos:pos+step_size]
        rows = replace_text_with_embedding(rows)
        pickle.dump(rows.values.squeeze().tolist(), ofile) # Transpose before storing for compatibility with pd.DataFrame() constructor when loading from pkl file
        rows = [] # Hacky way to free up memory :)
        pos += step_size
    
    if pos < len(df.index):
        rows = df.iloc[pos:]
        rows = replace_text_with_embedding(rows)
        pickle.dump(rows.values.squeeze().tolist(), ofile)
        rows = []

    ofile.close()

def fetch_embeddings_from_file(f, column_names, n_steps=500, get_df: bool = False) -> pd.DataFrame:
    """
    Function to fetch embeddings from file with given step size. Helps alleviate memory constraints with large embeddings sizes

    f: File object to fetch data from
    column_names: Name of columns of dataframe to be created
    n_rows: Amount of steps to fetch
    """
    
    data = [] #dict.fromkeys(column_names, []) #pd.DataFrame(columns=column_names)
    try:
        for _ in range(n_steps):
            rows_fetched = pickle.load(f)
            #print(rows_fetched)
            for row in rows_fetched:
                data.append(row)
        
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


if __name__ == "__main__":
    
    data_folder = Path(os.path.abspath(__file__)).parents[0] / "data" / "train_test"
    #hold_out_df = pd.read_csv(data_folder / "shooter_hold_out_test.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

    # Test compression and storing of embeddings
    test_df = pd.read_csv(data_folder / "shooter_hold_out_test.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    print(test_df)

    to_path = "test_pickle"
    create_and_store_embeddings(test_df, to_path, "glove", 3)


    # Decompress and load data into df
    ifile = bz2.BZ2File(to_path, "rb") # rb for read byte format
    loaded_df = fetch_embeddings_from_file(ifile, column_names=test_df.columns)
    print(loaded_df)
    #print(loaded_df.head())