from pathlib import Path
import os
import pandas as pd
from csv import QUOTE_NONE
from nltk.tokenize import RegexpTokenizer
import csv
import sys
from sklearn.model_selection import train_test_split

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


data_folder = Path(os.path.abspath("")) / "data"
target_path = Path(os.path.abspath("")) / "data" / "train_test_val"

df = pd.read_csv(data_folder / "all_labeled_split_512.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

print(target_path / "test.csv")


romano_df = df[df["name"] == "Jon Romano"]
robert_df = df[df["name"] == "Robert Butler jr"]

shooter_hold_out_test = pd.concat([romano_df, robert_df], ignore_index=True)
print(shooter_hold_out_test)

df = df[df["name"] != "Robert Butler jr"]
df = df[df["name"] != "Jon Romano"]

train_ratio = 0.8
val_ratio = 0.2

shooter_df = df[df["label"] == 1]
non_shooter_df = df[df["label"] == 0]

shooter_train, shooter_test = train_test_split(shooter_df, train_size=train_ratio)
non_shooter_train, non_shooter_test = train_test_split(non_shooter_df, train_size=train_ratio)

""" shooter_train, shooter_val = train_test_split(shooter_train, test_size=val_ratio)
non_shooter_train, non_shooter_val = train_test_split(non_shooter_train, test_size=val_ratio) """


final_train = pd.concat([shooter_train, non_shooter_train], ignore_index=True)
#final_val = pd.concat([shooter_val, non_shooter_val])

final_test = pd.concat([shooter_test, non_shooter_test], ignore_index=True)

final_train.to_csv(target_path / "train.csv", sep="‎", quoting=QUOTE_NONE, index=False)
#final_val.to_csv(target_path / "val.csv", sep="‎", quoting=QUOTE_NONE, index=False)
final_test.to_csv(target_path / "test.csv", sep="‎", quoting=QUOTE_NONE, index=False)
shooter_hold_out_test.to_csv(target_path / "shooter_hold_out_test.csv", sep="‎", quoting=QUOTE_NONE, index=False)