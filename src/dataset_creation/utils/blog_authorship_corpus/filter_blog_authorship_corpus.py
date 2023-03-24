from xml.etree import cElementTree as ET
from pathlib import Path
import shutil

blogs_folder = (Path(__file__).parents[3] / "data/blogs_authorship_corpus/blogs")
print(blogs_folder)

files = blogs_folder.glob("*.xml")
files = [file for file in files if (int(str(file).split(".")[2]) < 22 and int(str(file).split(".")[2]) > 12)]


filtered_blogs_path = (Path(__file__).parents[3] / "data/blogs_authorship_corpus/filtered_13-21_blogs")
for file in files:
    fname = file.stem
    shutil.move(file, (filtered_blogs_path/f"{fname}.xml"))