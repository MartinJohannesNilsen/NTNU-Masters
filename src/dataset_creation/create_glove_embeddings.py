import sys
import pandas as pd
import torch
from pathlib import Path
import os
from csv import QUOTE_NONE

sys.path.append("..")
from experiments.utils.word_embeddings import preprocess_text, get_glove_word_vectors

data_folder = Path(os.path.abspath(__file__)).parents[0] / "data"

train_df = pd.read_csv(data_folder / "train_test_val" / "train.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
test_df = pd.read_csv(data_folder / "train_test_val" / "test.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
hold_out_df = pd.read_csv(data_folder / "train_test_val" / "shooter_hold_out_test.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

def get_glove_emb(df: pd.df):
    df["text"] = df["text"].map(lambda a: preprocess_text(a)) # Preprocess into tokens to send to glove emb method
    df = df[df["text"].map(len) > 0] # Some entries from the previous step can become empty lists. Remove these

    df["text"] = df["text"].map(lambda a: get_glove_word_vectors(a, sentence_length=512, emb_dim=50))

    return df

train_df = get_glove_emb(train_df)
train_df.to_pickle(data_folder / "train_test_val" / "train_glove.pkl", compression="bz2")

test_df = get_glove_emb(test_df)
test_df.to_pickle(data_folder / "train_test_val" / "test_glove.pkl", compression="bz2")

hold_out_df = get_glove_emb(hold_out_df)
hold_out_df.to_pickle(data_folder / "train_test_val" / "hold_out_test_glove.pkl", compression="bz2")

""" df = pd.read_pickle(data_folder / "train_test_val" / "hold_out_glove.pkl", compression="bz2")
print(df) """