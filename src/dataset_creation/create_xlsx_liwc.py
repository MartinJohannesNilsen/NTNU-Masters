from csv import QUOTE_NONE, field_size_limit
from functools import partial
import os
from pathlib import Path
from typing import List
import click
import pandas as pd
import sys
field_size_limit(sys.maxsize)


def append_date_columns(dataframe: pd.DataFrame):
    df = dataframe.copy()
    date_dict = { "year": [], "month": [], "day": [] }
    for date in df["date"].convert_dtypes():
        year, month, day = [None, None, None] if type(date) == pd._libs.missing.NAType else (date.split("-") + 3 * [None])[:3]
        date_dict["year"].append(year)
        date_dict["month"].append(month)
        date_dict["day"].append(day)
    for key in date_dict.keys():
        df[key] = date_dict[key]

    # Need to format some of the columns to types
    df["date"] = pd.to_datetime(df['date'])
    # df["year"] = df["year"].astype("Int64")
    # df["month"] = df["month"].astype("Int64")
    # df["day"] = df["day"].astype("Int64")

    return df


def create_dictionary_of_dfs_from_paths(list_of_files: List[Path] = None, delimiter="‎"):
    assert list_of_files, "Need to define a list of paths to create dataframe from"

    # Create a dictionary of csv files per perpetrator
    paths = dict()
    for csv in list_of_files:
        parent = csv.parents[0]
        # key = parent.name
        key = str(parent.relative_to(parent.parents[1]))
        paths[key] = csv
    
    # Create a dictionary of dataframes per perpetrator
    dfs = dict()
    for key, value in paths.items():
        dfs[key] = pd.read_csv(value, delimiter=delimiter, engine='python', encoding="utf-8", quoting=QUOTE_NONE)
    
    return dfs

def _write_csv(df: pd.DataFrame, fname: str, out_dir: Path, create_dir_if_not_exists: bool = True, index: bool = False) -> bool:
    assert fname.split(".")[-1] == "csv", "Format needs to be csv"
    if create_dir_if_not_exists:
        out_dir.mkdir(parents=True, exist_ok=True) # Make sure the folder exists
    df.to_csv(out_dir / fname, sep="‎", header=True, index=False, date_format="%Y-%m-%d")

def _write_formatted_xlsx(df: pd.DataFrame, fname: str, out_dir: Path, create_dir_if_not_exists: bool = True, index: bool = False) -> bool:
    assert fname.split(".")[-1] == "xlsx", "Format needs to be excel"
    if create_dir_if_not_exists:
        out_dir.mkdir(parents=True, exist_ok=True) # Make sure the folder exists
    
    with pd.ExcelWriter(out_dir / fname, engine="xlsxwriter", datetime_format="DD-MM-YYYY") as writer:
        workbook = writer.book
        format = workbook.add_format({'text_wrap': True})
        df.to_excel(writer, sheet_name="Sheet1", index=False, na_rep='NaN')
        worksheet = writer.sheets['Sheet1']
        
        # Auto-adjust column width to max
        for column in df:
            column_length = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column)
            worksheet.set_column(col_idx, col_idx, column_length)
        
        # Some formatting for selected rows
        worksheet.set_column(0, 0, 6)
        worksheet.set_column(2, 2, 100, format)    


data_dir = Path(os.path.abspath(__file__)).parents[0] / "data"

def _store_xlsx(f_path):
    # Create path
    fname = Path(f_path).stem + ".xlsx"
    out_dir = data_dir / "xlsx"

    # Read data
    df = pd.read_csv(f_path, sep="‎", quoting=QUOTE_NONE, engine="python")

    # Write xlsx file
    _write_formatted_xlsx(df=df, fname=fname, out_dir=out_dir)

def main(path = data_dir / "train_test"):
    
    if os.path.isdir(path):
        for file in os.listdir(path):
            _store_xlsx(path / file)
    else:
        _store_xlsx(path)

if __name__ == "__main__":
    main()