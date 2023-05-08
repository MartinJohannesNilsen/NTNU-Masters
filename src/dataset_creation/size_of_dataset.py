import pandas as pd
from csv import QUOTE_NONE
import sys
import os
from pathlib import Path
import csv
from datetime import datetime

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

data_path = Path(os.path.abspath(__file__)).parents[0] / "data"

df  = pd.read_csv(str(data_path / "all_labeled.csv"), sep="‎", quoting=QUOTE_NONE, engine="python")

stair_twitter_df = df[df["name"] == "stair twitter archive"]
length_of_df = len(stair_twitter_df.index)
print(length_of_df)

stair_twitter_df["date"] = stair_twitter_df["date"].map(lambda a: make_datetime(a))
stair_twitter_drop_df = stair_twitter_df[stair_twitter_df["date"] < datetime(2016, 8, 13)]


df = df.drop(stair_twitter_drop_df.index, axis=0)

shooter_df = df[df["label"] == 1]
non_shooter_df = df[df["label"] == 0]
print(f"len whole dataset: {len(df.index)}")
print(f"len shooter set: {len(shooter_df.index)}")
print(f"len non-shooter set: {len(non_shooter_df.index)}")

all_df = 
