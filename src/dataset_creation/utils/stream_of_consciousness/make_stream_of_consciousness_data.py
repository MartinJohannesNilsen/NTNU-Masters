from xml.etree import cElementTree as ET
from pathlib import Path
import shutil
import pandas as pd


[print(_) for _ in Path(__file__).parents]
essays_folder = (Path(__file__).parents[4] / "data/stream_of_consciousness/essays.csv")

df_w_personality = pd.read_csv(essays_folder)
df_w_personality = df_w_personality.drop(columns=["#AUTHID"])
df_w_personality.insert(loc=0, column="date", value="")
df_w_personality.rename(columns={"date": "date", "TEXT": "text"}, inplace=True)
print(df_w_personality)

out_path_w_personality = (Path(__file__).parents[4] / "data/stream_of_consciousness/data_w_personality.csv")
out_path = (Path(__file__).parents[4] / "data/stream_of_consciousness/data.csv")

df_w_personality.to_csv(out_path_w_personality, sep="‎", index=False)

df = df_w_personality.drop(columns=["cEXT", "cNEU", "cAGR", "cCON", "cOPN"])
print(df)

df.to_csv(out_path, sep="‎", index=False)

df["name"] = "stream of consciousness"
print(df)
df.to_csv(Path(__file__).parents[2] / "data/stream_of_consciousness.csv", index=False, sep="‎")

""" out_path = (Path(__file__).parents[3] / "data/stream_of_consciousness/data.csv")
out_path_w_personality = (Path(__file__).parents[3] / "data/stream_of_consciousness/data_w_personality.csv")
df.to_csv(out_path, index=False, sep="‎") """