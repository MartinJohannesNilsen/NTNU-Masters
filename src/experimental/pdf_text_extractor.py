# importing required modules
from pathlib import Path, PureWindowsPath
from tika import parser
from PyPDF2 import PdfReader
import glob
import csv

root_folder = Path(__file__).resolve().parents[2]
src_folder = Path(__file__).resolve().parents[1]

doc_path = Path(Path(__file__).resolve().parents[2]/"schoolshootersinfo")
print(doc_path)
#doc_path = f"{doc_path}/**/*.pdf"
print(doc_path)

files = doc_path.glob("**/*")
print(files)

for file in files:
    reader = PdfReader(str(file.resolve()))
    fname = file.name
    fpath = src_folder/"raw_text"/fname

    with open(fpath, "w+") as f:
        for page in reader.pages:
            f.write(page.extract_text())
