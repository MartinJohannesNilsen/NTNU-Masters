import pandas as pd

df = pd.DataFrame([1,2,None,4], columns=["yo"])
print(df.index)
df = df[df["yo"].notna()]
print(df.index)