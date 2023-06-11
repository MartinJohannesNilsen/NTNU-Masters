import pandas as pd
from pathlib import Path
import os

data_dir = Path(os.path.abspath(__file__)).parents[4] / "data" / "raw_data" / "stream_of_consciousness"

df = pd.read_csv(data_dir / "essays.csv") # Read csv
df = df.drop(["#AUTHID", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"], axis=1) # Drop extra columns
df = df.rename(columns={'TEXT': 'text'}) # Rename text lowercase
df = df.assign(date="") # Assign date
df = df[['date', 'text']] # Reorder

df.to_csv(data_dir / "data.csv", sep="‎", header=True, index=False, date_format="%Y-%m-%d")