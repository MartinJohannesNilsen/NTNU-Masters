import sys
import pandas as pd
from pathlib import Path
import os
from csv import QUOTE_NONE
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from experiments.utils.word_embeddings import get_glove_word_vectors, get_fasttext_word_vectors, get_bert_word_embeddings


def replace_text_with_embedding(df: pd.DataFrame, method = "glove"):
    assert method == "glove" or method == "fasttext" or method == "bert", "Method not supported!"
    if method == "glove":
        df["text"] = df["text"].map(lambda a: get_glove_word_vectors(a, sentence_length=512))
    elif method == "fasttext":
        df["text"] = df["text"].map(lambda a: get_fasttext_word_vectors(a, sentence_length=512))
    elif method == "bert":
        df["text"] = df["text"].map(lambda a: get_bert_word_embeddings(a, sentence_length=512))
    else:
        return df
    
    df = df[df['text'].notna()]
    return df

# Load data
data_folder = Path(os.path.abspath(__file__)).parents[0] / "data" / "train_test"
train_df = pd.read_csv(data_folder / "train_no_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
test_df = pd.read_csv(data_folder / "test_no_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
hold_out_df = pd.read_csv(data_folder / "shooter_hold_out_test.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

# Create pickled embeddings
embeddings = ["glove", "fasttext", "bert"]
out_path = Path(os.path.abspath(__file__)).parents[1] / "experiments" / "features" / "embeddings"


for emb_type in embeddings:
    print(f"Type: {emb_type}")

    embedding_train_df = train_df.copy()
    embedding_train_df = replace_text_with_embedding(embedding_train_df, emb_type)
    embedding_train_df.to_pickle(out_path / f"train_no_stair_twitter_{emb_type}.pkl", compression="bz2")

    embedding_test_df = test_df.copy()
    embedding_test_df = replace_text_with_embedding(embedding_test_df, emb_type)
    embedding_test_df.to_pickle(out_path / f"test_no_stair_twitter_{emb_type}.pkl", compression="bz2")

    embedding_hold_out_df = hold_out_df.copy()
    embedding_hold_out_df = replace_text_with_embedding(embedding_hold_out_df, emb_type)
    embedding_hold_out_df.to_pickle(out_path / f"hold_out_test_{emb_type}.pkl", compression="bz2")