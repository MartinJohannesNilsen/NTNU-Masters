from functools import partial
from pathlib import Path
from typing import List
import pandas as pd

path = (Path(__file__).parents[3] / "data/twitter/")

smile_annotations = (path / "umass" / "umass_original.tsv")

df = pd.read_csv(smile_annotations, sep="\t")

df = df.loc[(df["Definitely English"] == 1) & (df["Automatically Generated Tweets"] == 0)]

df = df[["Date", "Tweet"]]
df = df.rename(columns={"Date":"date", "Tweet":"text"})

out_path = (Path(__file__).parents[3] / "src/dataset/outputs/umass_tweets.csv")
df.to_csv(out_path, index=False, sep="‎")