# importing required modules
from pathlib import Path
from PyPDF2 import PdfReader
import unicodedata
import re

root_folder = Path(__file__).resolve().parents[2]
src_folder = Path(__file__).resolve().parents[1]

# Path to pdfs containing school shooters' texts
doc_path = Path(Path(__file__).resolve().parents[2]/"schoolshootersinfo")

files = doc_path.glob("**/*.pdf")
print(f"Found these pdf files at {doc_path}:\n{files}")

for file in files:
    reader = PdfReader(str(file.resolve()))
    fname = file.stem
    fpath = src_folder/"raw_text"/f"{fname}.txt"

    with open(fpath, "w+", encoding="utf-8") as f:
        for page in reader.pages:
            text = page.extract_text()
            text = "".join(c if unicodedata.category(c) != "Co" else " " for c in text).strip() # Remove unwanted special chars and encode to utf-8
            text = re.sub(r'(-+)', '', text) # Some texts contain ----- as separators. Remove these separators
            f.write(text)
        