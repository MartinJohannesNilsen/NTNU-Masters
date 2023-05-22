import pandas as pd
from pathlib import Path
import os
from csv import QUOTE_NONE
import sys

base_path = Path(os.path.abspath(__file__)).parents[2] / "dataset_creation" / "data" / "train_test" / "new_preprocessed"
train_path = base_path / f"train_sliced_stair_twitter_256_preprocessed.csv"
val_path = base_path / f"val_sliced_stair_twitter_256_preprocessed.csv"
test_path = base_path / f"test_sliced_stair_twitter_256_preprocessed.csv"

train_df = pd.read_csv(train_path, sep="‎", quoting=QUOTE_NONE, engine="python")
val_df = pd.read_csv(val_path, sep="‎", quoting=QUOTE_NONE, engine="python")
test_df = pd.read_csv(test_path, sep="‎", quoting=QUOTE_NONE, engine="python")

base_path_liwc = Path(os.path.abspath(__file__)).parents[0] / "liwc" / "preprocessed" / "splits" / "csv" / "2022"
train_path_liwc = base_path_liwc / f"train_sliced_stair_twitter_256_preprocessed.csv"
val_path_liwc = base_path_liwc / f"val_sliced_stair_twitter_256_preprocessed.csv"
test_path_liwc = base_path_liwc / f"test_sliced_stair_twitter_256_preprocessed.csv"

train_liwc = pd.read_csv(train_path_liwc)
val_liwc = pd.read_csv(val_path_liwc)
test_liwc = pd.read_csv(test_path_liwc)

def check_coinciding(df, liwc_df):
    count = 0
    for i in range(len(liwc_df["text"])):
        original = str(df["text"].iloc[i+count])
        liwc = str(liwc_df["text"].iloc[i])

        if original.strip() != liwc.strip():
            print(f"Failed at row {i} with text columns:\nORIGINAL:\n{original}\nLIWC:\n{liwc}")
            count += 1

print("TESTING FOR 256")

print("Testing for preprocessed train")
check_coinciding(train_df, train_liwc)

print("Testing for preprocessed val")
check_coinciding(val_df, val_liwc)

print("Testing for preprocessed test")
check_coinciding(test_df, test_liwc)


print("TESTING FOR 512")

train_path = base_path / f"train_sliced_stair_twitter_512_preprocessed.csv"
val_path = base_path / f"val_sliced_stair_twitter_512_preprocessed.csv"
test_path = base_path / f"test_sliced_stair_twitter_512_preprocessed.csv"

train_df = pd.read_csv(train_path, sep="‎", quoting=QUOTE_NONE, engine="python")
val_df = pd.read_csv(val_path, sep="‎", quoting=QUOTE_NONE, engine="python")
test_df = pd.read_csv(test_path, sep="‎", quoting=QUOTE_NONE, engine="python")

train_path_liwc = base_path_liwc / f"train_sliced_stair_twitter_512_preprocessed.csv"
val_path_liwc = base_path_liwc / f"val_sliced_stair_twitter_512_preprocessed.csv"
test_path_liwc = base_path_liwc / f"test_sliced_stair_twitter_512_preprocessed.csv"

train_liwc = pd.read_csv(train_path_liwc)
val_liwc = pd.read_csv(val_path_liwc)
test_liwc = pd.read_csv(test_path_liwc)

print("Testing for preprocessed train")
check_coinciding(train_df, train_liwc)

print("Testing for preprocessed val")
check_coinciding(val_df, val_liwc)

print("Testing for preprocessed test")
check_coinciding(test_df, test_liwc)

for og, liwc in zip(train_df["text"], train_liwc["text"]):
    if len(str(liwc).strip()) == 1:
        print(og)
        print(liwc)
        print("\n")
