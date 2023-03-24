import pandas as pd

df = pd.read_csv("data/stream_of_consciousness/essays.csv") # Read csv
df = df.drop(["#AUTHID", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"], axis=1) # Drop extra columns
df = df.rename(columns={'TEXT': 'text'}) # Rename text lowercase
df = df.assign(date="") # Assign date
df = df[['date', 'text']] # Reorder

df.to_csv("data/stream_of_consciousness/data.csv", sep="‎", header=True, index=False, date_format="%Y-%m-%d")