import pandas as pd
from csv import QUOTE_NONE
import sys
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

import os
from pathlib import Path
from matplotlib import pyplot as plt
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

data_path = Path(os.path.abspath("")).parents[1] / "dataset_creation" / "data"

# Load data
df = pd.read_csv(data_path / "all_labeled.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
randy_twitter_drop_df = df[df["name"] == "stair twitter archive"]

def make_datetime(date):
        y, m, d = date.split("-")
        return datetime(int(y), int(m), int(d))

randy_twitter_drop_df["date"] = randy_twitter_drop_df["date"].map(lambda a: make_datetime(a))
randy_twitter_drop_df = randy_twitter_drop_df[randy_twitter_drop_df["date"] < datetime(2016, 8, 13)]
print(randy_twitter_drop_df)

df = df.drop(randy_twitter_drop_df.index, axis=0)

# Tokenize data

df["text"] = df["text"].map(lambda a: str(a).split(" "))

length_dict = {}

for t in df["text"]:
    post_len = len(t)
    if post_len < 100:
        if post_len in length_dict:
            length_dict[post_len] += 1
        else:
            length_dict[post_len] = 1
    
x = list(length_dict.keys())
x.sort()
y = list(length_dict.values())

plt.plot(x, y)
plt.title("Lengths of posts after preprocessing")
plt.xlabel("Length")
plt.ylabel("Count")
plt.show()
