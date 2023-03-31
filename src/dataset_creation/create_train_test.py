from pathlib import Path
import os
import pandas as pd
from csv import QUOTE_NONE
import csv
import sys
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

if __name__ == "__main__":
    _find_field_size_limit()

    # Set data folder
    data_folder = Path(os.path.abspath("")) / "data"
    target_path = Path(os.path.abspath("")) / "data" / "train_test"

    # Read data
    df = pd.read_csv(data_folder / "all_labeled_split_512.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

    # Hold out some shooters texts for testing purposes
    romano_df = df[df["name"] == "Jon Romano"]
    robert_df = df[df["name"] == "Robert Butler jr"]
    shooter_hold_out_test = pd.concat([romano_df, robert_df], ignore_index=True)
    df = df[df["name"] != "Robert Butler jr"]
    df = df[df["name"] != "Jon Romano"]

    # Divide into train and test, balanced 80% train of school shooters and not school shooters
    train_ratio = 0.8
    shooter_df = df[df["label"] == 1]
    non_shooter_df = df[df["label"] == 0]
    shooter_train, shooter_test = train_test_split(shooter_df, train_size=train_ratio)
    non_shooter_train, non_shooter_test = train_test_split(non_shooter_df, train_size=train_ratio)
    final_train = pd.concat([shooter_train, non_shooter_train], ignore_index=True)
    final_test = pd.concat([shooter_test, non_shooter_test], ignore_index=True)

    # Write dataframes to files
    final_train.to_csv(target_path / "train.csv", sep="‎", quoting=QUOTE_NONE, index=False)
    final_test.to_csv(target_path / "test.csv", sep="‎", quoting=QUOTE_NONE, index=False)
    shooter_hold_out_test.to_csv(target_path / "shooter_hold_out_test.csv", sep="‎", quoting=QUOTE_NONE, index=False)