from csv import QUOTE_NONE, field_size_limit
import os
from pathlib import Path
import pandas as pd
import sys
field_size_limit(sys.maxsize)
from functools import partial
import click

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


preprocessed = True
data_dir = Path(os.path.abspath(__file__)).parents[4] / "data" / "processed_data" / "train_test" / "csv" / f"{'preprocessed_glove' if preprocessed else ''}"

def _store_xlsx(f_path):
    # Create path
    fname = Path(f_path).stem + ".xlsx"
    out_dir = Path(os.path.abspath(__file__)).parents[4] / "data" / "processed_data" / "train_test" / "xlsx"

    # Read data
    df = pd.read_csv(f_path, sep="‎", quoting=QUOTE_NONE, engine="python")

    # Write xlsx file
    _write_formatted_xlsx(df=df, fname=fname, out_dir=out_dir)

def main(path = data_dir / "shooter_hold_out_256.csv"):
    
    if os.path.isdir(path):
        for file in os.listdir(path):
            _store_xlsx(path / file)
    else:
        _store_xlsx(path)

if __name__ == "__main__":
    main()