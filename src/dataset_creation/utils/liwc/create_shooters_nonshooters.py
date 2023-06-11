from csv import QUOTE_NONE
import os
from pathlib import Path
import pandas as pd

preprocessed = True
data_dir = Path(os.path.abspath(__file__)).parents[4] / "data" / "processed_data" / "train_test" / "csv" / f"{'preprocessed_glove' if preprocessed else ''}"
out_dir = Path(os.path.abspath(__file__)).parents[4] / "data" / "processed_data" / "shooters_nonshooters"

print(data_dir)
print(out_dir)


for max_len in ["256", "512"]:
    # Read in train, test and val
    train_df = pd.read_csv(data_dir / f"train_sliced_stair_twitter_{max_len}{'_preprocessed' if preprocessed else ''}.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    test_df = pd.read_csv(data_dir / f"test_sliced_stair_twitter_{max_len}{'_preprocessed' if preprocessed else ''}.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    val_df = pd.read_csv(data_dir / f"val_sliced_stair_twitter_{max_len}{'_preprocessed' if preprocessed else ''}.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

    # Combine
    all_df = pd.concat([train_df, test_df, val_df])

    # Filter on label
    shooters_df = all_df.loc[all_df['label'] == 1]
    nonshooters_df = all_df.loc[all_df['label'] == 0]

    # Save files
    os.makedirs(out_dir, exist_ok=True)
    shooters_df.to_csv(out_dir / f"shooters_{max_len}{'_preprocessed' if preprocessed else ''}.csv", sep="‎", header=True, index=False, date_format="%Y-%m-%d")
    nonshooters_df.to_csv(out_dir / f"nonshooters_{max_len}{'_preprocessed' if preprocessed else ''}.csv", sep="‎", header=True, index=False, date_format="%Y-%m-%d")