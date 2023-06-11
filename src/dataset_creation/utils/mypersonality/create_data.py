import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

def date_converter(date: str):
    if date is np.NaN:
        return ""
    obj = datetime.strptime(date, '%m/%d/%y %I:%M %p')
    return obj.strftime('%Y-%m-%d')

if __name__ == "__main__":
    data_dir = Path(os.path.abspath(__file__)).parents[4] / "data" / "raw_data" / "mypersonality"

    df = pd.read_csv(data_dir / "mypersonality_small.csv") # Read csv
    df = df.drop(["#AUTHID","sEXT","sNEU","sAGR","sCON","sOPN","cEXT","cNEU","cAGR","cCON","cOPN","NETWORKSIZE","BETWEENNESS","NBETWEENNESS","DENSITY","BROKERAGE","NBROKERAGE","TRANSITIVITY"], axis=1) # Drop extra columns
    df = df.rename(columns={'STATUS': 'text', 'DATE': 'date'}) # Rename columns
    df['date'] = df['date'].apply(date_converter) # Assign date
    df = df[['date', 'text']] # Reorder
    df["name"] = "mypersonality"

    df.to_csv(data_dir / "data.csv", sep="‎", header=True, index=False, date_format="%Y-%m-%d")