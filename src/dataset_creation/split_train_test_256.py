from pathlib import Path
import os
import pandas as pd
from csv import QUOTE_NONE
import csv
import sys
from datetime import datetime


def _find_field_size_limit():
    max_int = sys.maxsize
    while True:
        # Decrease the value by factor 10 as long as the OverflowError occurs.
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int/10)


def split_rows(df: pd.DataFrame, token_lim: int, trunc_threshold: int = 20):
    """
    Method to split long posts into smaller, easier to process posts.

    df: Pandas dataframe object consisting at least of a row names "text"
    token_lim: Max amount of words/tokens to be allowed per row
    trunc_threshold: The minimum amount of tokens left in new row to not truncate
    """


    new_rows = []
    for _, row in df.iterrows():

        tokens = row["text"].split(" ")
        tokens_left = len(tokens)
        threshold = 20 # Threshold to determine split or truncation

        if tokens_left > token_lim:
            lower = 0
            upper = token_lim - 1

            while tokens_left > threshold:
                new_row = [row["date"], tokens[lower:upper], row["name"], row["label"]]
                new_rows.append(new_row)
                lower += token_lim
                upper += token_lim
                tokens_left -= token_lim

        else:
            new_rows.append([row["date"], tokens, row["name"], row["label"]])

    # Create new dataframe
    new_df = pd.DataFrame(new_rows, columns=df.columns)
    new_df["text"] = new_df["text"].map(lambda a: " ".join(a))

    return new_df


if __name__ == "__main__":
    _find_field_size_limit()

    # Read all labeled data
    data_path = Path(os.path.abspath(__file__)).parent / "data" / "train_test"
    
    train_df = pd.read_csv(data_path / "train_no_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    test_df = pd.read_csv(data_path / "test_no_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    shooter_hold_out_df = pd.read_csv(data_path / "shooter_hold_out_test.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

    train_df = split_rows(train_df, 256)
    test_df = split_rows(test_df, 256)
    shooter_hold_out_df = split_rows(shooter_hold_out_df, 256)
    

    # Write dataframe to csv
    train_df.to_csv(data_path / "train_no_stair_twitter_256.csv", sep="‎", quoting=QUOTE_NONE, index=False)           
    test_df.to_csv(data_path / "test_no_stair_twitter_256.csv", sep="‎", quoting=QUOTE_NONE, index=False)           
    shooter_hold_out_df.to_csv(data_path / "shooter_hold_out_test_256.csv", sep="‎", quoting=QUOTE_NONE, index=False)           