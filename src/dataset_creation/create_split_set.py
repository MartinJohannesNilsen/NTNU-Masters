from pathlib import Path
import os
import pandas as pd
from csv import QUOTE_NONE
from nltk.tokenize import RegexpTokenizer
import csv
import sys

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

df = pd.read_csv(data_folder / "all_labeled.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

new_rows = []
for _, row in df.iterrows():
    tokens = row["text"].split(" ")
    tokens_left = len(tokens)
    threshold = 20 # Threshold to determine split or truncation
    doc_size = 512

    if tokens_left > doc_size:
        lower = 0
        upper = doc_size - 1

        while tokens_left > threshold:
            new_row = [row["date"], tokens[lower:upper], row["name"], row["label"]]
            new_rows.append(new_row)
            lower += doc_size
            upper += doc_size
            tokens_left -= doc_size

    else:
        new_rows.append([row["date"], tokens, row["name"], row["label"]])

new_df = pd.DataFrame(new_rows, columns=df.columns)
new_df["text"] = new_df["text"].map(lambda a: " ".join(a))

new_df.to_csv(data_folder / "all_labeled_split_512.csv", sep="‎", quoting=QUOTE_NONE, index=False)