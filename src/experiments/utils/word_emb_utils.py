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



def tokenize_with_preprocessing_drop_len(text: str, remove_url: bool = True):
    """
    Convenience function to perform tokenization, but drop the length output of original tokenization function.

    Args:
        text (str): Text sequence to be tokenized
        remove_url (bool, optional): Whether to remove URL completely or replace with URLHYPERLINK. Defaults to True.

    Returns:
        out (list(str)): List of tokens
    """
    out, _ = tokenize_with_preprocessing(text, remove_url=remove_url)
    return out


def _split_safe(text):
    return str(text).split(" ")


                                        ### EMBEDDINGS ###

def get_emb_model(model_name: str):
    """
    Convenience function to get embedding model object

    Args:
        model_name (str): Name of embedding model. Can be glove, glove_50, fasttext or bert

    Returns:
        embedding model: Return the desired embedding model
    """

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

def embed_text(text:str, emb_model, emb_type: str):
    """
    Create word embeddings from text passed to method and return a list of embedding vectors along with the length of each sentence before padding

    Args:
        text (str): Text sequence to be converted into word embeddings
        emb_model: Embedding model to be used
        emb_type (str): Name of embedding model

    Returns:
        Tuple(tensor, int): Tuple containing a tensor of word embeddings and the length of the sequence of the list
    """

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


def pad_embeddings(embeddings: torch.tensor, max_len: int, pad_pos: str):
    """
    Perform padding to max_len needed to use the embedding vectors as input to a neural network architecture.

    Args:
        embeddings (tensor): Tensor of word embeddings also containing the original length of only word embeddings
        max_len (int): Max length of tensor
        pad_pos (str): Position to apply padding

    Returns:
        Tuple(tensor, int): 0-padded tensor containing the original word embeddings and the sequence length before padding.  
    """

    emb_dim = embeddings.shape[1]
    seq_len = embeddings.shape[0]

    if embeddings.shape[0] < max_len: # Check if padding is needed
        req_padding = max_len - embeddings.shape[0]
        pad_tensor = torch.zeros(req_padding, emb_dim)

        if pad_pos == "head":
            # Pad at beginning of tensor
            embeddings = torch.cat((pad_tensor, embeddings), dim=0)

        elif pad_pos == "tail":
            # Pad at end of tensor
            embeddings = torch.cat((embeddings, pad_tensor), dim=0)

        else:
            # Pad 50/50 at front and end of tensor
            split_i = floor(pad_tensor.shape[0]/2)
            embeddings = torch.cat((pad_tensor[:split_i], embeddings), dim=0)
            embeddings = torch.cat((embeddings, pad_tensor[split_i:]), dim=0)
            
    # If tensor passed to method is larger than max_len -> truncate to max_len
    elif embeddings.shape[0] > max_len:
        embeddings = embeddings[:max_len, :]
        seq_len = max_len

    return embeddings, seq_len


def embed_and_pad(text: str, emb_model, max_len: int, pad_pos: str, emb_type: str):
    """
    Convenience function to do both embedding and padding in one method.

    Args:
        text (str): Text sequence to be processed
        emb_model: Embedding model to be used
        max_len (int): Max allowed length of text sequence
        pad_pos (str): Position to apply padding
        emb_type (str): Name of embedding type

    Returns:
        Tuple(tensor, int): Tuple containing a padded tensor of word embeddings and the original length of the text sequence before padding 
    """
    res = embed_text(text, emb_model, emb_type)
    if not res:
        return
    
    embs, _ = res
    embs, seq_len = pad_embeddings(embs, max_len=max_len, pad_pos=pad_pos)
    
    return embs, seq_len

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