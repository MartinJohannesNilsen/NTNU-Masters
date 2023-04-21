import time
from typing import Union
import pandas as pd
from pathlib import Path
from csv import QUOTE_NONE
import os
import sys
import torch
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from utils.word_embeddings import get_glove_word_vectors, get_fasttext_word_vectors, get_bert_word_embeddings, _tokenize_with_preprocessing, _apply_fixed_sentence_length
import h5py
import numpy as np
from bs4 import BeautifulSoup
import torch
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from torchtext.vocab import FastText, GloVe
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
import re
from typing import List
from transformers import pipeline, logging
import os
from pathlib import Path
from math import floor

data_path = Path(os.path.abspath("")).parents[1] / "dataset_creation" / "data" / "train_test"

# Load data
t = time.time()
df = pd.read_csv(data_path / "train_sliced_stair_twitter.csv", sep="‎", quoting=QUOTE_NONE, engine="python")[:500]
print(f"loading data: {time.time()-t} s")

# Tokenize data
t = time.time()
df["text"] = df["text"].map(lambda a: _tokenize_with_preprocessing(a))
print(f"tokenizing data: {time.time()-t} s")

cache_dir = Path(os.path.abspath(__file__)).parents[3] / "resources" / ".vector_cache"

# Get embeddings
emb_dim = 300
name = "840B"

t = time.time()
glove_vec = GloVe(name=name, dim=emb_dim, cache=cache_dir)
print(f"loading glove dict: {time.time()-t} s")


def _extract_embeddings(tokenized_input):

    if len(tokenized_input) == 0:
        return
    
    res = glove_vec.get_vecs_by_tokens(tokenized_input, lower_case_backup=True)
        
    return res

t = time.time()
df["text"] = df["text"].map(lambda a: _extract_embeddings(a))
print(f"extracting embeddings: {time.time()-t} s")


t = time.time()
df["text"] = df["text"].map(lambda a: _apply_fixed_sentence_length(a, sentence_length=512, emb_dim=300, pad_pos="tail"))
print(f"applying padding: {time.time()-t} s")
