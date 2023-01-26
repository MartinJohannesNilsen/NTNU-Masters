# importing required modules
from pathlib import Path

doc_path = Path(__file__).resolve().parents[2] / "schoolshootersinfo"
print("Path:", doc_path)

files = (doc_path).rglob("*.pdf")
for path in files:
    print(path)
