from csv import QUOTE_NONE, field_size_limit
from functools import partial
import os
from pathlib import Path
from typing import List
import click
import pandas as pd
import sys
from pathlib import Path
from csv import field_size_limit

max_int = sys.maxsize
while True:
    # Decrease the value by factor 10 as long as the OverflowError occurs.
    try:
        field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int/10)

# Convenience script file to combine all previously created csv file into a singular dataset

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

    df["date"] = pd.to_datetime(df['date'])

    return df


def create_dictionary_of_dfs_from_paths(list_of_files: List[Path] = None, delimiter="‎"):
    assert list_of_files, "Need to define a list of paths to files to create dataframe from"

    # Create a dictionary of csv files per perpetrator
    paths = dict()
    for csv_file in list_of_files:
        parent = csv_file.parents[0]
        key = str(parent.relative_to(parent.parents[1]))
        paths[key] = csv_file
    
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
        

LABEL_DICT = {
    "school_shooters": 1,
    "stair_twitter_archive": 1,
    "stream_of_consciousness": 0,
    "mypersonality": 0,
    "twitter": 0,
}
    

click.option = partial(click.option, show_default=True)
@click.command()
@click.option("-d", "--dataset", type=click.Choice(["school_shooters", "stair_twitter_archive", "stream_of_consciousness", "mypersonality", "twitter", "all"]), default="school_shooters", help="Folder to create dataframe from")
@click.option("-v", "--verbose", type=click.IntRange(0, 2), default=1, help="Verbosity for prints to terminal")
@click.option("-o", "--out_dir", default=(Path(os.path.abspath(__file__)).parents[2] / "data" / "processed_data"), help="Save to folder")
@click.option("-f", "--format", type=click.Choice(["csv", "xlsx"]), default="csv", help="Save to spreadsheet")
@click.option("-l", "--labeled", is_flag=True, default=True, help="Create a file of all datasets with labels")
def main(dataset, verbose, out_dir, format, labeled):

    """ if verbose > 0:
        print("Creating dataframe from folder:", dataset)
        if verbose > 1 and dataset != "all":
            print(f"Full path: {Path(os.path.abspath(__file__)).parents[0] / 'data' / dataset}") """
    data_dir = Path(os.path.abspath(__file__)).parents[2] / "data" / "raw_data"
    print(data_dir)
    # Create dictionary of dataframes
    if dataset == "all":
        dfs = create_dictionary_of_dfs_from_paths(data_dir.rglob("data.csv"))
    else:
        dfs = create_dictionary_of_dfs_from_paths((data_dir / dataset).rglob("data.csv"))
        
    # Append year, month and day columns, while creating a list of dataframes 
    list_of_names = list(dfs.keys())
    # list_of_dfs = [append_date_columns(df).assign(name=list_of_names[index].replace("_", " ")) for index, df in enumerate(dfs.values())]
    if labeled:
        list_of_dfs = [append_date_columns(df).assign(name=str(Path(list_of_names[index].replace("_", " ")).name), label=LABEL_DICT[str(Path(list_of_names[index]).parent) if str(Path(list_of_names[index]).parent) != "raw_data" else str(Path(list_of_names[index]).name)]) for index, df in enumerate(dfs.values())]
    else:
        list_of_dfs = [append_date_columns(df).assign(name=str(Path(list_of_names[index].replace("_", " ")).name)) for index, df in enumerate(dfs.values())]
    

    # Create one big dataframe
    try:
        df = pd.concat(list_of_dfs).reset_index()
    except ValueError:
        print("No data!")
        exit(1)
    
    # Print stats
    if verbose > 1:
        print("-"*40)
        print(df.info())
        print("-"*40)    
    
    # Create folder and define file name
    out_dir = Path(out_dir)
    fname = f"{dataset}_labeled.{format}" if labeled else f"{dataset}.{format}"

    # Print if verbose
    if verbose > 0:
        print("Saving to file:", f"{out_dir.name}/{fname}")
    
    # Create file
    # Drop year, month, day from excel-dataframe
    if format == "xlsx":
        excel_df = df.drop(["year", "month", "day"], axis=1)
        _write_formatted_xlsx(df=excel_df, fname=fname, out_dir=out_dir)
    else:
        # df = df.drop(["index", "year", "month", "day", "name"], axis=1)
        df = df.drop(["index", "year", "month", "day"], axis=1) # want to include name
        _write_csv(df=df, fname=fname, out_dir=out_dir)

if __name__ == "__main__":
    main()