import numpy as np
import pandas as pd
from datetime import datetime

def date_converter(date: str):
    if date is np.NaN:
        return ""
    obj = datetime.strptime(date, '%m/%d/%y %I:%M %p')
    return obj.strftime('%Y-%m-%d')

if __name__ == "__main__":
    df = pd.read_csv("data/mypersonality/mypersonality_small.csv") # Read csv
    df = df.drop(["#AUTHID","sEXT","sNEU","sAGR","sCON","sOPN","cEXT","cNEU","cAGR","cCON","cOPN","NETWORKSIZE","BETWEENNESS","NBETWEENNESS","DENSITY","BROKERAGE","NBROKERAGE","TRANSITIVITY"], axis=1) # Drop extra columns
    df = df.rename(columns={'STATUS': 'text', 'DATE': 'date'}) # Rename columns
    df['date'] = df['date'].apply(date_converter) # Assign date
    df = df[['date', 'text']] # Reorder

    df.to_csv("data/mypersonality/data.csv", sep="‎", header=True, index=False, date_format="%Y-%m-%d")