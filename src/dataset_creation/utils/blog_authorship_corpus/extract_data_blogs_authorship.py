from pathlib import Path
import pandas as pd
from lxml import etree
import re

month_dict = {
    'january': '01',
    'february': '02',
    'march': '03',
    'april': '04',
    'may': '05',
    'june': '06',
    'july': '07',
    'august': '08',
    'september': '09',
    'october': '10',
    'november': '11',
    'december': '12',
    'janvier': '01',
    'février': '02',
    'mars': '03',
    'avril': '04',
    'mai': '05',
    'juin': '06',
    'juillet': '07',
    'août': '08',
    'septembre': '09',
    'octobre': '10',
    'novembre': '11',
    'décembre': '12',
    'gennaio': '01',
    'febbraio': '02',
    'marzo': '03',
    'aprile': '04',
    'maggio': '05',
    'giugno': '06',
    'luglio': '07',
    'agosto': '08',
    'settembre': '09',
    'ottobre': '10',
    'novembre': '11',
    'dicembre': '12',
    'enero': '01',
    'febrero': '02',
    'marzo': '03',
    'abril': '04',
    'mayo': '05',
    'junio': '06',
    'julio': '07',
    'agosto': '08',
    'septiembre': '09',
    'octubre': '10',
    'noviembre': '11',
    'diciembre': '12',
    'janeiro': '01',
    'fevereiro': '02',
    'março': '03',
    'abril': '04',
    'maio': '05',
    'junho': '06',
    'julho': '07',
    'setembro': '09',
    'outubro': '10',
    'novembro': '11',
    'dezembro': '12',
    'januar': '01',
    'februar': '02',
    'mars': '03',
    'april': '04',
    'mai': '05',
    'juni': '06',
    'juli': '07',
    'august': '08',
    'september': '09',
    'oktober': '10',
    'november': '11',
    'desember': '12',
    'januari': '01',
    'februari': '02',
    'maart': '03',
    'mei': '05',
    'augustus': '08',
    'styczeń': '01',
    'luty': '02',
    'marzec': '03',
    'kwiecień': '04',
    'maj': '05',
    'czerwiec': '06',
    'lipiec': '07',
    'sierpień': '08',
    'wrzesień': '09',
    'październik': '10',
    'listopad': '11',
    'grudzień': '12',
    'yanvar': '01',
    'fevral': '02',
    'mart': '03',
    'aprel': '04',
    'may': '05',
    'iyun': '06',
    'iyul': '07',
    'avgust': '08',
    'sentabr': '09',
    'oktabr': '10',
    'noyabr': '11',
    'dekabr': '12'
}


blogs_path = (Path(__file__).parents[3] / "data/blogs_authorship_corpus/filtered_13-21_blogs")
out_path = (Path(__file__).parents[3] / "data/blogs_authorship_corpus/blogs_csv")
files = blogs_path.glob("*.xml")

clean_string = lambda s: re.sub(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]+', '', s)

for file in files:
        
    xml_text = ""
    with open(file, mode="r", errors="ignore") as f:
        xml_text = clean_string(f.read())

    # Parse XML tree structure
    tree = etree.fromstring(xml_text, parser=etree.XMLParser(recover=True))

    posts_list = []

    # Parse all post tags and append to dataframe
    for post, date in zip(tree.xpath("//post"), tree.xpath("//date")):
        date_list = date.text.split(",")
        formatted_date = f"{date_list[2]}-{month_dict[str(date_list[1]).lower()]}-{date_list[0]}"
        
        post_text = post.text.strip()

        posts_list.append([formatted_date, post_text])

    # Write to file
    out_file = out_path / f"{file.stem}.csv"
    df = pd.DataFrame(posts_list, columns=["date", "text"])
    df.to_csv(out_file, sep="‎", index=False)
    



