import sys
import pandas as pd
from pathlib import Path
import os
from csv import QUOTE_NONE
sys.path.append("..")
from experiments.utils.word_embeddings import preprocess_text, get_glove_word_vectors

data_folder = Path(os.path.abspath(__file__)).parents[0] / "data"

train_df = pd.read_csv(data_folder / "train_test" / "train.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
test_df = pd.read_csv(data_folder / "train_test" / "test.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
hold_out_df = pd.read_csv(data_folder / "train_test" / "shooter_hold_out_test.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

def get_glove_emb(df: pd.DataFrame):
    df["text"] = df["text"].map(lambda utterance: preprocess_text(utterance)) # Preprocess into tokens to send to glove emb method
    df = df[df["text"].map(len) > 0] # Some entries from the previous step can become empty lists. Remove these

    df["text"] = df["text"].map(lambda utterance: get_glove_word_vectors(utterance, sentence_length=512, size_small=False))

    return df

train_df = get_glove_emb(train_df)
train_df.to_pickle(data_folder / "train_test" / "train_glove.pkl", compression="bz2")

test_df = get_glove_emb(test_df)
test_df.to_pickle(data_folder / "train_test" / "test_glove.pkl", compression="bz2")

hold_out_df = get_glove_emb(hold_out_df)
hold_out_df.to_pickle(data_folder / "train_test" / "hold_out_test_glove.pkl", compression="bz2")