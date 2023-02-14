from functools import partial
from pathlib import Path
from typing import List
import click
import pandas as pd


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
    return df


def create_dictionary_of_dfs_from_paths(list_of_files: List[Path] = None):
    assert list_of_files, "Need to define a list of paths to create dataframe from"

    # Create a dictionary of csv files per perpetrator
    paths = dict()
    for csv in list_of_files:
        paths[csv.parents[0].name] = csv
    
    # Create a dictionary of dataframes per perpetrator
    dfs = dict()
    for key, value in paths.items():
        dfs[key] = pd.read_csv(value, delimiter="‎ ", engine='python', encoding="utf-8")
    
    return dfs
    

click.option = partial(click.option, show_default=True)
@click.command()
@click.option("-d", "--dataset", type=click.Choice(["schoolshootersinfo", "masshooters", "manifestos", "all"]), default="schoolshootersinfo", help="Folder to create dataframe from")
@click.option("-v", "--verbose", type=click.IntRange(0, 2), default=1, help="Verbosity for prints to terminal")
@click.option("-s", "--save", is_flag=True, help="Save to spreadsheet")
def main(dataset, verbose, save):

    if verbose > 0:
        print("Creating dataframe from folder:", dataset)

    # Create dictionary of dataframes
    if dataset == "all":
        dfs = create_dictionary_of_dfs_from_paths((Path(__file__).parents[2] / "data").rglob("*.csv"))
    else:
        dfs = create_dictionary_of_dfs_from_paths((Path(__file__).parents[2] / "data" / dataset).rglob("*.csv"))
    
    # Append year, month and day columns, while creating a list of dataframes 
    list_of_names = list(dfs.keys())
    list_of_dfs = [append_date_columns(df).assign(name=list_of_names[index].replace("_", " ")) for index, df in enumerate(dfs.values())]
    
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
    
    if save:
        # Create folder and define file name
        out_dir = (Path(__file__).parent / "outputs")
        out_dir.mkdir(parents=True, exist_ok=True) # Make sure the folder exists
        fname = f"{dataset}.xlsx" 
        
        # Print if verbose
        if verbose > 0:
            print("Saving to file:", f"{out_dir.name}/{fname}")
        
        # Create file
        df.to_excel(out_dir / fname, index=False)

if __name__ == "__main__":
    main()