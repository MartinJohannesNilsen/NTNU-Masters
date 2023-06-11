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
    """
    Helper function to combat issues with csv field sizes in Windows.
    Reduces max field size value by factor 10, starting at sys.maxsize, until it no longer overflows
    """
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int/10)


def make_datetime(date: str):
    """
    Converts a given date in string format to actual datetime.datetime object
    
    Args:
        date (str): A date given as a string in the format y-m-d
    
    Returns:
        datetime object (datetime.datetime(y,m,d))
    """

    y, m, d = date.split("-")
    return datetime(int(y), int(m), int(d))


def make_split_df(df: pd.DataFrame, max_len: int):
    """
    Splits each element in dataframe based on given max length of each "text" element in dataframe.
    If new row resulting from a split is under a given threshold (20), drop that row.

    Args:
        df (pandas.DataFrame): Dataframe consisting of at least columns [date, text, name, label]
        max_len (int): Max length of a text element.

    Returns:
        new_df (pandas.DataFrame): A new dataframe with columns [date, text, name, label] with rows split based on provided max_len argument
    """

    new_rows = []
    for _, row in df.iterrows():

        tokens = row["text"].split(" ")
        tokens_left = len(tokens)
        threshold = 20 # Threshold to determine split or truncation

        if tokens_left > max_len:
            lower = 0
            upper = max_len - 1

            # Iterate through text element in row. Split text element if larger than given max_len
            # Continue until length of resulting split text element < max_len
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


def make_train_test_val(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.4):
    """
    Create train, test, val splits from given dataframe. Each split has same ratio of shooters and non-shooters.

    Args:
        df (pandas.DataFrame): Dataframe containing columns [date, text, name, label]
        train_ratio (float): Ratio of samples to take from total dataframe to create train split
        val_ratio (float): Ratio of samples to take from test split to create val split

    Returns:
        final_train (pandas.DataFrame): Dataframe containing the train split of the total data
        final_test (pandas.DataFrame): Dataframe containing the test split of the total data
        final_val (pandas.DataFrame): Dataframe containing the val split of the total data
    """

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
    """
    Convenience function to write 512 splits and 256 splits of train, test, val to file

    Args:
        df (pandas.DataFrame): Dataframe containing complete dataset
        stair_str (str): String denoting if randy stair archive is sliced or removed completely. Can be sliced_stair or no_stair

    Returns:
        void 
    """
    
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
    data_folder = Path(os.path.abspath(__file__)).parent / "data"
    out_folder = data_folder / "train_test" / "new"

    task = "train_test_split"

    if task == "train_test_split":
        df = pd.read_csv(data_folder / "original" / "all_labeled.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
        randy_twitter_df = df[df["name"] == "stair twitter archive"]
        randy_twitter_df["date"] = randy_twitter_df["date"].map(lambda a: make_datetime(a))
        randy_twitter_drop_df = randy_twitter_df[randy_twitter_df["date"] < datetime(2016, 8, 13)]

        df = df.drop(randy_twitter_drop_df.index, axis=0)
        write_to_file(df, "sliced_stair")

        df = pd.read_csv(data_folder / "original" / "all_labeled.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
        df = df.drop(randy_twitter_df.index, axis=0)
        write_to_file(df, "no_stair")
    
    elif task == "shooter_hold_out":
        df = pd.read_csv(data_folder / "original" / "shooter_hold_out.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
        split_256_df = make_split_df(df.copy(), 256)
        split_256_df.to_csv(out_folder / "shooter_hold_out_256.csv", sep="‎", quoting=QUOTE_NONE, index=False)
        split_512_df = make_split_df(df.copy(), 512)
        split_512_df.to_csv(out_folder / "shooter_hold_out_512.csv", sep="‎", quoting=QUOTE_NONE, index=False)

    elif task == "randy_stair_set":
        df = pd.read_csv(data_folder / "original" / "all_labeled.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
        randy_twitter_df = df[df["name"] == "stair twitter archive"]
        randy_twitter_df["date"] = randy_twitter_df["date"].map(lambda a: make_datetime(a))
        randy_twitter_drop_df = randy_twitter_df[randy_twitter_df["date"] < datetime(2016, 8, 13)]
        final_randy_df = randy_twitter_df.drop(randy_twitter_drop_df.index, axis=0)

        out_folder = Path(os.path.abspath(__file__)).parents[2] / "data" / "stair_twitter_archive"
        final_randy_df.to_csv(out_folder / "final_used_stair_tweets.csv", sep="‎", quoting=QUOTE_NONE, index=False)
