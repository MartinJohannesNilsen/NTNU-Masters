from pathlib import Path
import os
import pandas as pd
from csv import QUOTE_NONE
import csv
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split 

# Maxsize of csv field size
def _find_field_size_limit():
    max_int = sys.maxsize
    while True:
        # Decrease the value by factor 10 as long as the OverflowError occurs.
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int/10)


def make_datetime(date):
        y, m, d = date.split("-")
        return datetime(int(y), int(m), int(d))


def make_split_df(df, max_len: int):
    new_rows = []
    for _, row in df.iterrows():

        tokens = row["text"].split(" ")
        tokens_left = len(tokens)
        threshold = 20 # Threshold to determine split or truncation

        if tokens_left > max_len:
            lower = 0
            upper = max_len - 1

            while tokens_left > threshold:
                new_row = [row["date"], tokens[lower:upper], row["name"], row["label"]]
                new_rows.append(new_row)
                lower += max_len
                upper += max_len
                tokens_left -= max_len

        else:
            new_rows.append([row["date"], tokens, row["name"], row["label"]])

    # Create new dataframe
    new_df = pd.DataFrame(new_rows, columns=df.columns)
    new_df["text"] = new_df["text"].map(lambda a: " ".join(a))

    return new_df


def make_train_test_val(df, train_ratio: int = 0.8, val_ratio: int = 0.4):

    shooter_df = df[df["label"] == 1]
    non_shooter_df = df[df["label"] == 0]

    shooter_train, shooter_test = train_test_split(shooter_df, train_size=train_ratio)
    shooter_val, shooter_test = train_test_split(shooter_test, train_size=val_ratio)

    non_shooter_train, non_shooter_test = train_test_split(non_shooter_df, train_size=train_ratio)
    non_shooter_val, non_shooter_test = train_test_split(non_shooter_test, train_size=val_ratio)

    final_train = pd.concat([shooter_train, non_shooter_train], ignore_index=True)
    final_test = pd.concat([shooter_test, non_shooter_test], ignore_index=True)
    final_val = pd.concat([shooter_val, non_shooter_val], ignore_index=True)

    return final_train, final_test, final_val

def write_to_file(df, stair_str):
    
    split_512_df = make_split_df(df.copy(), 512)
    train, test, val = make_train_test_val(split_512_df)
    train.to_csv(out_folder / f"train_{stair_str}_twitter_512.csv", sep="‎", quoting=QUOTE_NONE, index=False)
    test.to_csv(out_folder / f"test_{stair_str}_twitter_512.csv", sep="‎", quoting=QUOTE_NONE, index=False)
    val.to_csv(out_folder / f"val_{stair_str}_twitter_512.csv", sep="‎", quoting=QUOTE_NONE, index=False)
    split_512_df, train, test, val = None, None, None, None

    split_256_df = make_split_df(df.copy(), 256)
    train, test, val = make_train_test_val(split_256_df)
    train.to_csv(out_folder / f"train_{stair_str}_twitter_256.csv", sep="‎", quoting=QUOTE_NONE, index=False)
    test.to_csv(out_folder / f"test_{stair_str}_twitter_256.csv", sep="‎", quoting=QUOTE_NONE, index=False)
    val.to_csv(out_folder / f"val_{stair_str}_twitter_256.csv", sep="‎", quoting=QUOTE_NONE, index=False)
    split_256_df, train, test, val = None, None, None, None

if __name__ == "__main__":
    _find_field_size_limit()

    # Read all labeled data
    data_folder = Path(os.path.abspath("")) / "data"
    out_folder = data_folder / "train_test" / "new"

    df = pd.read_csv(data_folder / "all_labeled.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    randy_twitter_df = df[df["name"] == "stair twitter archive"]
    randy_twitter_df["date"] = randy_twitter_df["date"].map(lambda a: make_datetime(a))
    randy_twitter_drop_df = randy_twitter_df[randy_twitter_df["date"] < datetime(2016, 8, 13)]

    df = df.drop(randy_twitter_drop_df.index, axis=0)
    write_to_file(df, "sliced_stair")

    # No stair twitter
    df = pd.read_csv(data_folder / "all_labeled.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    df = df.drop(randy_twitter_df.index, axis=0)
    write_to_file(df, "no_stair")
