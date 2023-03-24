from functools import partial
from pathlib import Path
from typing import List
import pandas as pd

path = (Path(__file__).parents[3] / "data/twitter/")

smile_annotations = (path / "smileannotationsfinal.csv")

df = pd.read_csv(smile_annotations, names=["id", "text", "emotion"])

df = df.loc[(df["emotion"] != "nocode") & (df["emotion"] != "not-relevant")]

df = pd.DataFrame(df["text"])
df.insert(0, 'date', "")

out_path = (Path(__file__).parents[3] / "src/dataset/outputs/smileannotated_correct_format.csv")

df.to_csv(out_path, index=False, sep="‎")