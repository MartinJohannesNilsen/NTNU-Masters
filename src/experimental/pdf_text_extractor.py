# importing required modules
from pathlib import Path, PureWindowsPath
from tika import parser
import glob

#print(__file__)

doc_path = Path(f"{__file__}/../../schoolshootersinfo").resolve()
print(doc_path)
doc_path = f"{doc_path}/**/*.pdf"
print(doc_path)

files = glob.glob(doc_path, recursive=True)
print(files)

for file in files:
    print(file)
