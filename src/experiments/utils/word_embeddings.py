# Imports
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
import numpy as np
import pandas as pd

cache_dir = Path(os.path.abspath(__file__)).parents[3] / "resources" / ".vector_cache"

def _pad_and_get_orig_seq_len(text, max_len):
    
    og_len = len(text)

    if og_len < max_len:
        req_padding = max_len - og_len
        [text.append(float(0)) for _ in range(req_padding)]
    
    elif og_len > max_len:
        text = text[:max_len]

    return text, og_len

def _apply_fixed_sentence_length(embedding: torch.tensor, sentence_length: int, emb_dim: int, pad_pos: str = "tail") -> torch.tensor:
    """Pad if embedding is smaller then sentence length, and truncate if longer.

    Args:
        embedding (torch.tensor): Embedding from the torch.vocab or transformers pipeline functions.
        sentence_length (int): Fixed sentence length.
        emb_dim (int): The last dimensionality value.
        pad_pos (str): The position to apply padding, head / tail / split (head and tail split)

    Returns:
        torch.tensor: New tensor with fixed sentence length.
    """

    if embedding.shape[0] < sentence_length:
        req_padding = sentence_length - embedding.shape[0]
        pad_tensor = torch.zeros(req_padding, emb_dim)

        if pad_pos == "head":
            embedding = torch.cat((pad_tensor, embedding), dim=0)

        elif pad_pos == "tail":
            embedding = torch.cat((embedding, pad_tensor), dim=0)

        else:
            split_i = floor(pad_tensor.shape[0]/2)
            embedding = torch.cat((pad_tensor[:split_i], embedding), dim=0)
            embedding = torch.cat((embedding, pad_tensor[split_i:]), dim=0)
            

    elif embedding.shape[0] > sentence_length:
        embedding = embedding[:sentence_length, :]

    

    return embedding

def get_seq_len(seq):
    """
    Due to the way embeddings were stored at the beginning of the project, extracting lengths of the individual sequences was deemed necessary
    """
    whole_seq_len = len(seq)

    i = 1
    while i < whole_seq_len:
        if np.any(seq[-i]): # Start at last element in tensor and work backwards
            break

        i += 1
        
    return whole_seq_len - (i-1)


def _tokenize_with_preprocessing(text: str, remove_url: bool = True):
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

    return cleaned_words


def get_bert_word_embeddings(input: str or List[str], pretrained_name = "bert-base-uncased", sentence_length: int = None, pad_pos: str = "tail"):
    """Generates BERT word embeddings, using the Transformers pipeline. NB! The batch dimension is squeezed out as default.

    Args:
        input (str or List[str]): Input string or list of input strings.
        pretrained_name (str, optional): Which pre-trained model to use. Defaults to 'bert-base-uncased'.
        chunk_size (int, optional): The size of chunks. Padding will be applied. Defaults to 512.
        pad_pos (str, optional): Defines where to pad the tensor if tensor is too short. Defaults to padding at the end --> "tail"

    Returns:
        torch.tensor or List[torch.tensor]: If input is string, returns a tensor of dimensions (n_tokens, emb_dim=768). 
                                            If input is a list of strings, returns a list of tensors with dimensions (n_tokens, emb_dim=768).
                                            Note that n_tokens will be equal to sentence_length if defined.

    """
    logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = AutoModel.from_pretrained(pretrained_name)
    extract_features = pipeline('feature-extraction', model=model, tokenizer=tokenizer, padding=True, truncation=True)
    
    # If string: dim (1, n_tokens, 768)
    # If List[str]: dim (n_entries, 1, n_tokens, 768)
    # NB! The dimension of 1 is going to be removed for similarity to the other embeddings
    out = extract_features(input) 
    emb_dim = 768

    # For either a single string, or list of strings: 
    # - Removes the batch dimension (1)
    # - Converts to tensor
    # - Applies fixed sentence length if defined
    if isinstance(input, str):
        return _apply_fixed_sentence_length(torch.tensor(out).squeeze(), sentence_length=sentence_length, emb_dim=emb_dim, pad_pos=pad_pos) if sentence_length else torch.tensor(out).squeeze()
    else:
        return [_apply_fixed_sentence_length(torch.tensor(e).squeeze(), sentence_length=sentence_length, emb_dim=emb_dim, pad_pos=pad_pos) for e in out] if sentence_length else [torch.tensor(e).squeeze() for e in out]


def get_glove_model(size_small: bool = True):
    emb_dim = 50 if size_small else 300
    name = "6B" if size_small else "840B"

    glove_vec = GloVe(name=name, dim=emb_dim, cache=cache_dir)

    return glove_vec
    

def get_emb_matrix(emb_dim, emb_type, vocab_len, word_to_index):

    path = None
    if emb_type == "glove":
        dim_name = "6B.50d" if emb_dim == 50 else "840B.300d"
        path = cache_dir / f"glove.{dim_name}.txt"
    else:
        path = cache_dir / "wiki.en.vec"

    #embed_mat = np.random.rand(vocab_len + 1, emb_dim)
    embed_mat = np.zeros([vocab_len + 1, emb_dim])

    with open(path, "r+", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            elem = line.split(" ")
            word = elem[0]

            if word not in word_to_index:
                continue

            word_idx = word_to_index[word]

            if word_idx <= vocab_len:
                embed_mat[word_idx] = np.asarray(elem[1:], dtype='float32')

    return embed_mat

def split_safe(text):
    return str(text).split(" ")

def create_vocab_w_idx(df: pd.DataFrame, is_preprocessed: bool = True):
    """
    Input:
        df: Pandas dataframe containing a 'text' column

    Output:
        dict: A dictionary containing all unique words in vocab and their counts
    """
    if not is_preprocessed:
        df["text"] = df["text"].map(lambda a: _tokenize_with_preprocessing(a))


    counts = {}
    for row in df["text"].map(lambda a: split_safe(a)):
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
    for token in split_safe(text):
        if token in word_to_idx:
            ids.append(word_to_idx[token])
        else:
            ids.append(1)

    return ids


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
            
    elif length > sentence_length:
        ids = ids[:max_len]

    return np.array(ids), length


def get_padded_ids(text, word_to_idx, pad_pos, max_len):
    ids = get_id_from_tokens(text, word_to_idx)

    padded_ids, length = pad_ids(ids, pad_pos, max_len)

    return [padded_ids, length]


def get_glove_word_vectors(input: str or List[str], emb_model = None, sentence_length: int = None, emb_dim: int = 300, pad_pos: str = "tail"):
    """Generates word vectors in the format of GloVe, using torch.vocab.

    Args:
        input (str or List[str]): Input string or list of input strings.
        size_small (bool, optional): If False, use the 2.18GB pre-trained model instead of the 862MB one. Defaults to True. Emb_dim is 50 for small, and 300 for large.
        sentence_length (int, optional): If defined, pads the list of tokens to desired length. Truncates if longer. Defaults to None.
        pad_pos (str, optional): Defines where to pad the tensor if tensor is too short. Defaults to padding at the end --> "tail"

    Returns:
        torch.tensor or List[torch.tensor]: If input is string, returns a tensor of dimensions (n_tokens, emb_dim=50 or 300). 
                                            If input is a list of strings, returns a list of tensors with dimensions (n_tokens, emb_dim=50 or 300).
                                            Note that n_tokens will be equal to sentence_length if defined.
    """
    def _extract_embeddings(input):
        tokenized_input = _tokenize_with_preprocessing(input)

        if len(tokenized_input) == 0:
            return

        res = emb_model.get_vecs_by_tokens(tokenized_input, lower_case_backup=True)
        
        res = _apply_fixed_sentence_length(res, sentence_length=sentence_length, emb_dim=emb_dim, pad_pos=pad_pos)
        
        return res

    if isinstance(input, str):
        return _extract_embeddings(input)
    else:
        return [_extract_embeddings(text) for text in input]


def get_ft_model():
    return FastText(language="en", cache=cache_dir) # 6.6GB



def get_fasttext_word_vectors(input: str or List[str], emb_model = None, sentence_length: int = None, pad_pos: str = "tail"):
    """Generates word vectors in the format of FastText, using torch.vocab.

    Args:
        input (str or List[str]): Input string or list of input strings.
        sentence_length (int, optional): If defined, pads the list of tokens to desired length. Truncates if longer. Defaults to None.
        pad_pos (str, optional): Defines where to pad the tensor if tensor is too short. Defaults to padding at the end --> "tail"

    Returns:
        torch.tensor or List[torch.tensor]: If input is string, returns a tensor with dimensions (n_tokens, emd_dim=300). 
                                            If input is a list of strings, returns a list of tensors with dimensions (n_tokens, emd_dim=300).
    """

    def _extract_embeddings(input):
        tokenized_input = _tokenize_with_preprocessing(input)

        if len(tokenized_input) == 0:
            return
        
        vec = emb_model # 6.6GB
        res = vec.get_vecs_by_tokens(tokenized_input, lower_case_backup=True)
        
        emb_dim = 300
        res = _apply_fixed_sentence_length(res, sentence_length=sentence_length, emb_dim=emb_dim, pad_pos=pad_pos)
        
        return res

    if isinstance(input, str):
        return _extract_embeddings(input)
    else:
        return [_extract_embeddings(text) for text in input]


if __name__ == "__main__":
    # Text
    example1 = "It does not do to dwell on dreams and forget to live, remember that. Now, why don’t you put that admirable Cloak back on and get off to bed?"
    example2 = "Just because you’ve got the emotional range of a teaspoon doesn’t mean we all have."
    
    # Test plural (List[str])
    # bert = get_bert_word_embeddings([example1, example2]) # List with 2 entries, tensors of dim (39, 768) and (24, 768)
    # glove = get_glove_word_vectors([example1, example2]) # List with 2 entries, tensors of dim (11, 50) and (5, 50)
    # fasttext = get_fasttext_word_vectors([example1, example2]) # List with 2 entries, tensors of dim (11, 300) and (5, 300)
    
    # Test singular (str)
    # bert = get_bert_word_embeddings(example1) # Tensor of dim (39, 768)
    # glove = get_glove_word_vectors(example1) # Tensor of dim (11, 50)
    # fasttext = get_fasttext_word_vectors(example1) # Tensor of dim (11, 300)
    
    # Test defined sentence length
    # bert = get_bert_word_embeddings(example1, sentence_length=5) # Tensor of dim (5, 768)
    glove = get_glove_word_vectors(example1, sentence_length=256, size_small=False, pad_pos="split") # Tensor of dim (5, 50)
    # fasttext = get_fasttext_word_vectors(example1, sentence_length=5) # Tensor of dim (5, 300)

    print(glove.shape)
    print(glove)

    for row in glove.numpy():
        if sum(row) != 0:
            print(row)