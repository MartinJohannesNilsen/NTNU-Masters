import sys 
import os
from pathlib import Path
import pandas as pd

#from word_embeddings import *
import torch

from csv import QUOTE_NONE
import sys
import csv

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


base_path = Path(os.path.abspath(__file__)).parents[2] / "dataset_creation" / "data"
datasets = {
    "school_shooters": base_path / "school_shooters.csv",
    "manifestos": base_path / "manifestos.csv",
    "stair_twitter_archive": base_path / "stair_twitter_archive.csv",
    "twitter": base_path / "twitter.csv",
}

schoolshootersinfo_df = pd.read_csv(datasets["school_shooters"], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)
#manifesto_df = pd.read_csv(datasets["manifestos"], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)
stair_twitter_archive_df = pd.read_csv(datasets["stair_twitter_archive"], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)
twitter_df = pd.read_csv(datasets["twitter"], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)

schoolshootersinfo_df["shooter"] = 1
#manifesto_df["shooter"] = 1
stair_twitter_archive_df["shooter"] = 1
twitter_df["shooter"] = 0

print(schoolshootersinfo_df.text.str.len().max())
#print(manifesto_df.text.str.len().max())
print(stair_twitter_archive_df.text.str.len().max())
print(twitter_df.text.str.len().max())

from bs4 import BeautifulSoup
import string
import re
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
import nltk
from word_embeddings import preprocess_text, get_glove_word_vectors
import itertools
import seaborn as sns
from sklearn.model_selection import train_test_split 

#nltk.download()

shooter_df = pd.concat([schoolshootersinfo_df, stair_twitter_archive_df])[:100]
non_shooter_df = twitter_df[:100]
shooter_df["text"].map(lambda a: preprocess_text(a))
non_shooter_df["text"].map(lambda a: preprocess_text(a))


shooter_words = list(itertools.chain.from_iterable(shooter_df["text"]))
non_shooter_words = list(itertools.chain.from_iterable(non_shooter_df["text"]))
#print(non_shooter_words)

shooter_word_freqs = nltk.FreqDist(shooter_words)
non_shooter_word_freqs = nltk.FreqDist(non_shooter_words)

whole_corpus_df = pd.concat([shooter_df, non_shooter_df]).sample(frac=1)
whole_corpus_df["text"].map(lambda a: get_glove_word_vectors(a))

x_train, x_test, y_train, y_test = train_test_split(whole_corpus_df["text"], whole_corpus_df["shooter"], test_size=0.1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

print(x_train)

""" sns.set_style("darkgrid")
shooter_word_freqs.plot(20)
non_shooter_word_freqs.plot(20) """


