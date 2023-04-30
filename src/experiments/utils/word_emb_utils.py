# Imports
from bs4 import BeautifulSoup
import torch
from transformers import pipeline, logging, AutoTokenizer, AutoModel
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from torchtext.vocab import FastText, GloVe
import re
import os
from pathlib import Path
from math import floor
import numpy as np
import pandas as pd


cache_dir = Path(os.path.abspath(__file__)).parents[3] / "resources" / ".vector_cache"


# PREPROCESSING

def tokenize_with_preprocessing(text: str, remove_url: bool = True):
    """For the embedders needing tokenized input. The method perform certain steps of text cleaning:
        - Stopword removal
        - Url replacement (token or removal)
        - Username removal
        - Hashtag removal
        - Character normalization

    Args:
        text (str): Input string.
        remove_url (bool, optional): Removes url completely. Defaults to True.

    Returns:
        List[str]: List of tokens
    """

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    tokenizer = RegexpTokenizer("[\w']+")
    words = tokenizer.tokenize(text)

    url_replacement = "" if remove_url else "URLHYPERLINK"

    cleaned_words = []
    for word in words:
        word = word.lower()
        if word not in stopwords.words("english"):
            word = re.sub(r'http+|www+', url_replacement, word) # Replace urls with chosen string or remove completely
            word = re.sub(r'@[^ ]+', '', word) # Remove usernames in the context of Twitter posts
            word = re.sub(r'#', '', word) # Remove hashtags and keep words
            #word = re.sub(r'[^a-zA-Z0-9\s]', '', word) # ADDED AFTER WE MADE EMBS!!!!! Remove more special chars
            word = re.sub(r'([A-Za-z])\1{2,}', r'\1', word) # Character normalization, prevent words with letters repeated more than twice

            if word != "":
                cleaned_words.append(word)

    return cleaned_words, len(cleaned_words)


# EMBEDDING

def get_emb_model(model_name: str):
    model = None

    if model_name == "glove":
        model = GloVe(name="840B", dim=300, cache=cache_dir)
    if model_name == "glove_50":
        model = GloVe(name="6B", dim=50, cache=cache_dir)
    if model_name == "fasttext":
        model = FastText(language="en", cache=cache_dir)

    if model_name == "bert":
        logging.set_verbosity_error()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        b_model = AutoModel.from_pretrained("bert-base-uncased")
        model = pipeline('feature-extraction', model=b_model, tokenizer=tokenizer, padding=True, truncation=True)
    
    return model


def embed_text(text:str, emb_model):

    embs, tokens_len = None, None

    # Bert processing and feature extraction
    if isinstance(emb_model, FeatureExtractionPipeline):
        embs = torch.tensor(emb_model(text)).squeeze()
        tokens_len = embs.shape[0]

    # Glove and fasttext
    else:
        processed_text, tokens_len = tokenize_with_preprocessing(text, remove_url=True)
       
        # If preprocessing ends up removing whole text, stop
        if len(processed_text) == 0:
            return
        
        embs = emb_model.get_vecs_by_tokens(processed_text, lower_case_backup=True)

    return embs, tokens_len


def pad_input(embeddings, max_len: int, pad_pos: str):
    emb_dim = embeddings.shape[1]

    if embeddings.shape[0] < max_len:
        req_padding = max_len - embeddings.shape[0]
        pad_tensor = torch.zeros(req_padding, emb_dim)

        if pad_pos == "head":
            embeddings = torch.cat((pad_tensor, embeddings), dim=0)

        elif pad_pos == "tail":
            embeddings = torch.cat((embeddings, pad_tensor), dim=0)

        else:
            split_i = floor(pad_tensor.shape[0]/2)
            embeddings = torch.cat((pad_tensor[:split_i], embeddings), dim=0)
            embeddings = torch.cat((embeddings, pad_tensor[split_i:]), dim=0)
            

    elif embeddings.shape[0] > max_len:
        embeddings = embeddings[:max_len, :]

    return embeddings


def embed_and_pad(text: str, emb_model, max_len: int, pad_pos: str):
    res = embed_text(text, emb_model)
    if not res:
        return
    
    embs, seq_len = res
    embs = pad_input(embs, max_len=max_len, pad_pos=pad_pos)
    
    return embs, seq_len


if __name__ == "__main__":
    model = get_emb_model("bert")

    test = "This is a test txt"
    embs, length = embed_text(test, model)
    print(embs)
    print(length)
    print(test)

    """ model = get_emb_model("glove")
    embs, length = embed_text(test, model)
    print(embs)
    print(length)

    model = get_emb_model("fasttext")
    embs, length = embed_text(test, model)
    print(embs)
    print(length) """

    print(pad_input(embs, 9, "head"))