from pathlib import Path
import os
import pandas as pd
from csv import QUOTE_NONE
import csv
import sys

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
    data_dir = Path(os.path.abspath(__file__)).parents[2] / "data" / "processed_data" / "train_test" / "csv"
    all_fpaths = data_dir.rglob("*.csv")
    emb_type = "glove"

    def tokenize_and_throw_len(text: str, emb_type: str = "glove"):
        """
        Helper function to throw length argument of tokenizing function.

        Args:
            text (str)
            emb_type (str): Embedding type to tokenize for. Can be fasttext or glove

        Returns:
            processed_text (List(str)): List of tokens after preprocessing
        """
        processed_text, _ = tokenize_with_preprocessing(text, emb_type=emb_type)

        return processed_text

    # Loop through all files and preprocess
    dir_str = "preprocessed_glove" if emb_type == "glove" else "preprocessed_ft"
    dest_dir = Path(os.path.abspath(__file__)).parents[2] / "data" / "processed_data" / "train_test"
    for fpath in all_fpaths:
        print(fpath)
        df = pd.read_csv(fpath, sep="‎", quoting=QUOTE_NONE, engine="python")
        df["text"] = df["text"].map(lambda a: " ".join(tokenize_and_throw_len(a, emb_type="glove")))
        stem = Path(fpath).stem
        df.to_csv(str(dest_dir / dir_str / stem) + "_preprocessed.csv", sep="‎", index=False)