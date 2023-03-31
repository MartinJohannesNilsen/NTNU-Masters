from pathlib import Path
import os
import pandas as pd
from csv import QUOTE_NONE
import csv
import sys

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

    # Read all labeled data
    data_folder = Path(os.path.abspath("")) / "data"
    df = pd.read_csv(data_folder / "all_labeled.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

    # Split or truncate all posts based on fixed maximum doc_size
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

    # Create new dataframe
    new_df = pd.DataFrame(new_rows, columns=df.columns)
    new_df["text"] = new_df["text"].map(lambda a: " ".join(a))

    # Write dataframe to csv
    new_df.to_csv(data_folder / "all_labeled_split_512.csv", sep="‎", quoting=QUOTE_NONE, index=False)