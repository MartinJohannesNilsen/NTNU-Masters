import os
from pathlib import Path

parent_dir = Path(__file__).parents[0]
fpaths = parent_dir.rglob("*")

for file in fpaths:
    new_fname = str(file).replace(":","-")
    os.rename(file, new_fname)
    print(new_fname)