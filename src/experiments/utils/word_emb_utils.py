# Imports
from bs4 import BeautifulSoup
import torch
from transformers import pipeline, logging, AutoTokenizer, AutoModel
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from torchtext.vocab import FastText, GloVe
from nltk import word_tokenize
import re
import os
from pathlib import Path
from math import floor
import numpy as np
import pandas as pd


cache_dir = Path(os.path.abspath(__file__)).parents[3] / "resources" / ".vector_cache"


                                        ### PREPROCESSING ###

def tokenize_with_preprocessing(text: str, remove_url: bool = True, emb_type: str = "glove"):
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

    url_replacement = "" if remove_url else "URLHYPERLINK"

    cleaned_words = []
    for word in text.split(" "):
        word = word.lower()
        if word not in stopwords.words("english"):
            word = re.sub(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})", url_replacement, word) # Replace urls with chosen string or remove completely
            word = re.sub(r'@[^ ]+', '', word) # Remove usernames in the context of Twitter posts
            word = re.sub(r'#', '', word) # Remove hashtags and keep words
            word = re.sub(r'([A-Za-z])\1{2,}', r'\1', word) # Character normalization, prevent words with letters repeated more than twice
            word = re.sub(r'([^\w\s\'`])\1+', r'\1', word)

            if word != "":
                cleaned_words.append(word)
    
    cleaned_words = RegexpTokenizer("[\w']+").tokenize(" ".join(cleaned_words)) if emb_type != "fasttext" else word_tokenize(" ".join(cleaned_words))

    return cleaned_words, len(cleaned_words)



def tokenize_with_preprocessing_drop_len(text, remove_url: bool = True):
    out, _ = tokenize_with_preprocessing(text, remove_url=remove_url)
    return out


def _split_safe(text):
    return str(text).split(" ")


                                        ### EMBEDDINGS ###

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


# UTILS TO PRE-COMPUTE EMBEDDINGS FOR LATER STORAGE

def embed_text(text:str, emb_model, emb_type):

    embs, tokens_len = None, None

    # Bert processing and feature extraction
    if isinstance(emb_model, FeatureExtractionPipeline):
        embs = torch.tensor(emb_model(text)).squeeze()
        tokens_len = embs.shape[0]

    # Glove and fasttext
    else:
        processed_text, tokens_len = tokenize_with_preprocessing(text, remove_url=True, emb_type=emb_type)
       
        # If preprocessing ends up removing whole text, stop
        if len(processed_text) == 0:
            return
        
        embs = emb_model.get_vecs_by_tokens(processed_text, lower_case_backup=True)

    return embs, tokens_len


def pad_embeddings(embeddings, max_len: int, pad_pos: str):
    emb_dim = embeddings.shape[1]
    seq_len = embeddings.shape[0]

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
        seq_len = max_len

    return embeddings, seq_len


def embed_and_pad(text: str, emb_model, max_len: int, pad_pos: str, emb_type: str):
    res = embed_text(text, emb_model, emb_type)
    if not res:
        return
    
    embs, _ = res
    embs, seq_len = pad_embeddings(embs, max_len=max_len, pad_pos=pad_pos)
    
    return embs, seq_len


# UTILS FOR LIVE EMBEDDING AND CREATION OF EMBEDDING LAYERS

def pad_ids(ids, pad_pos, max_len):
    length = len(ids)
    if length < max_len:
        req_padding = max_len - length
        pad = [0 for _ in range(req_padding)]

        if pad_pos == "head":
            pad += ids
            ids = pad

        elif pad_pos == "tail":
            ids += pad

        else:
            split_i = floor(req_padding/2)
            front_pad = [0 for _ in range(split_i)]
            end_pad = [0 for _ in range(req_padding - split_i)]
            front_pad += ids
            front_pad += end_pad
            ids = front_pad
            
    elif length > max_len:
        ids = ids[:max_len]

    return np.array(ids), length


def get_emb_matrix(emb_dim, emb_type, vocab_len, word_to_index):

    path = None
    if emb_type == "glove_50":
        path = cache_dir / "glove.6B.50d.txt"
    elif emb_type == "glove":
        path = cache_dir / "glove.840B.300d.txt"
    else:
        path = cache_dir / "wiki.en.vec"

    embed_mat = np.zeros([vocab_len + 1, emb_dim])

    with open(path, "r+", encoding="utf-8") as f:
        for line in f:
            elem = line.split(" ")
            word = elem[0]

            if word not in word_to_index:
                continue

            word_idx = word_to_index[word]

            if word_idx <= vocab_len:
                embed_mat[word_idx] = np.asarray(elem[1:], dtype='float32')

    return embed_mat


def create_vocab_w_idx(df: pd.DataFrame, is_preprocessed: bool = True):
    """
    Input:
        df: Pandas dataframe containing a 'text' column

    Output:
        dict: A dictionary containing all unique words in vocab and their counts
    """
    if not is_preprocessed:
        df["text"] = df["text"].map(lambda a: tokenize_with_preprocessing_drop_len(a))

    counts = {}
    for row in df["text"].map(lambda a: _split_safe(a)):
        for word in row:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1

    # Drop underrepresented words and create vocab
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for key in counts.keys():
        if counts[key] >= 3:
            vocab[key] = len(vocab)

    return vocab


def get_id_from_tokens(text, word_to_idx):

    ids = []
    for token in _split_safe(text):
        if token in word_to_idx:
            ids.append(word_to_idx[token])
        else:
            ids.append(1)

    return ids


def tokenize_and_pad(text, word_to_idx, pad_pos, max_len):
    
    ids = get_id_from_tokens(text, word_to_idx)
    padded_ids, length = pad_ids(ids, pad_pos, max_len)

    return [padded_ids, length]


if __name__ == "__main__":
    model = get_emb_model("bert")

    test = "This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul This is a test txt jonas er kul"
    embs, length = embed_and_pad(test, model, max_len=6, pad_pos="tail")
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

    #print(pad_embeddings(embs, 9, "head"))