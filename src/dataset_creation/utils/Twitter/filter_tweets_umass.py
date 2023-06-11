from functools import partial
from pathlib import Path
from typing import List
import pandas as pd

data_dir = Path(os.path.abspath(__file__)).parents[4] / "data" / "raw_data" / "twitter" / "umass"

df = pd.read_csv(data_dir / "umass_original.tsv", sep="\t")

df = df.loc[(df["Definitely English"] == 1) & (df["Automatically Generated Tweets"] == 0)]

df = df[["Date", "Tweet"]]
df = df.rename(columns={"Date":"date", "Tweet":"text"})

df.to_csv(data_dir / "umass_only_english.csv", index=False, sep="‎")