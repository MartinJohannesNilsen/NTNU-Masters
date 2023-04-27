import pandas as pd
from word_embeddings import _tokenize_with_preprocessing
from pathlib import Path
from csv import QUOTE_NONE
import sys
import os



def create_vocab_w_idx(df: pd.DataFrame, is_preprocessed: bool = True):
    """
    Input:
        df: Pandas dataframe containing a 'text' column

    Output:
        dict: A dictionary containing all unique words in vocab and their counts
    """
    if not is_preprocessed:
        df["text"] = df["text"].map(lambda a: _tokenize_with_preprocessing(a))

    def split_safe(text):
        return str(text).split(" ")

    counts = {}
    for row in df["text"].map(lambda a: split_safe(a)):
        for word in row:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1

    # Drop underrepresented words and create vocab
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for key in counts.keys():
        if counts[key] >= 3:
            vocab[key] = len(vocab)

    return vocab


if __name__ == "__main__":

    base_path = Path(os.path.abspath(__file__)).parents[2] / "dataset_creation" / "data" / "train_test_preprocessed"

    df1 = pd.read_csv(base_path / "shooter_hold_out_test_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    df2 = pd.read_csv(base_path / "test_sliced_stair_twitter_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    df3 = pd.read_csv(base_path / "train_sliced_stair_twitter_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

    df = pd.concat([df1,df2,df3], axis=0)
    print(df)

    word_2_index = create_vocab_w_idx(df, is_preprocessed=True)

    print(word_2_index)