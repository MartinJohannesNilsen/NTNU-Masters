# importing required modules
from pathlib import Path
import fitz

# Opening document
doc = fitz.open(Path(__file__).parents[2] / 'schoolshootersinfo/William_Atchison/documents/atchison_online_1.0.pdf')

# Extracting text
for page in doc:
    text = page.get_text()
    print(text)
