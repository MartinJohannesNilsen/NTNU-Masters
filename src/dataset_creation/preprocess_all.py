from pathlib import Path
import os
import pandas as pd
from csv import QUOTE_NONE
import csv
import sys
from datetime import datetime

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from experiments.utils.word_emb_utils import tokenize_with_preprocessing

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
    data_folder = Path(os.path.abspath("")) / "data" / "train_test" / "new"
    dest_folder = Path(os.path.abspath("")) / "data" / "train_test" / "new_preprocessed"

    all_fpaths = data_folder.rglob("*.csv")

    def tokenize_and_throw_len(text):
        processed_text, _ = tokenize_with_preprocessing(text)
        return processed_text

    for fpath in all_fpaths:
        df = pd.read_csv(fpath, sep="‎", quoting=QUOTE_NONE, engine="python")
        df["text"] = df["text"].map(lambda a: " ".join(tokenize_and_throw_len(a)))
        stem = Path(fpath).stem
        df.to_csv(str(dest_folder / stem) + "_preprocessed.csv", sep="‎", index=False)