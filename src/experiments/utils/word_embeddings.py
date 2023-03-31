# Imports
import os
from pathlib import Path
import numpy as np
import torch
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from torchtext.vocab import FastText, GloVe
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup

import re
from typing import List
from transformers import pipeline

def get_bert_word_embeddings(input, pretrained_name = "bert-base-uncased", to_list: bool = True):
    """A method for generating BERT word embeddings using a pre-trained tokenizer. Returns result as either torch tensors (default) or regular lists. 

    Args:
        input (str or List[str]): Input string or list of input strings.
        chunk_size (int, optional): The size of chunks. Padding will be applied. Defaults to 512.
        pretrained_tokenizer (str, optional): Which pre-trained tokenizer to use. Defaults to 'bert-base-uncased'.
        do_lower_case (bool, optional): If True, lowercasing will be applied. Defaults to False.
        to_list (bool, optional): If True, returns a 2d array with the chunked lists. Defaults to False.

    Returns:
        list or torch.tensor: A 2d array of chunked word embeddings. Defaults to torch.tensor.
    """

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = AutoModel.from_pretrained(pretrained_name)

    pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
    out = pipe(input)

    # if embedding_type(input) is str:
    if isinstance(input, str):
        if to_list: return out[0]
        else: return torch.tensor(out[0])
    else:
        arr = [e[0] for e in out]
        if to_list: return arr
        else: return torch.tensor(arr)
       
def preprocess_text(text: str, full_clean_url: bool = True):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    tokenizer = RegexpTokenizer("[\w']+")
    words = tokenizer.tokenize(text)

    url_replacement = "" if full_clean_url else "URLHYPERLINK"

    cleaned_words = []
    for word in words:
        word = word.lower()
        if word not in stopwords.words("english"):
            word = re.sub(r'http+|www+', url_replacement, word) # Replace urls with chosen string or remove completely
            word = re.sub(r'@[^ ]+', '', word) # Remove usernames in the context of Twitter posts
            word = re.sub(r'#', '', word) # Remove hashtags and keep words
            word = re.sub(r'([A-Za-z])\1{2,}', r'\1', word) # Character normalization, prevent words with letters repeated more than twice

            if word != "":
                cleaned_words.append(word)

    #words = [re.sub(r'http+|www+', url_replacement, word).lower() for word in words if word not in stopwords.words("english")] 

    #return [word for word in words if word != ""]
    return cleaned_words

    
def get_glove_word_vectors(words: List[List[str]], sentence_length: int, size_small: bool = True, to_list: bool = False, emb_dim: int = 50):
    """Generates word vectors in the format of GloVe, using torch.vocab.

    Args:
        text (str): Input string.
        size_small (bool, optional): If False, use the 2.18GB pre-trained model instead of the 862MB one. Defaults to True.
        to_list (bool, optional): If True, returns a 2d array with the word vectors. Defaults to False.

    Returns:
        list or torch.tensor: A 2D array containing all the word vectors. Defaults to torch.tensor.
    """
    # vec = GloVe(name='6B', dim=50) # 862MB
    #vec = GloVe(name='840B', dim=300) # 2.18GB
    glove_vec = GloVe(name='6B', dim=50) if emb_dim == 50 else GloVe(name='840B', dim=300)
    res = glove_vec.get_vecs_by_tokens(words, lower_case_backup=True)
    # Pad tensor if needed
    if res.shape[0] < sentence_length:
        try:
            req_padding = sentence_length - res.shape[0]
            pad_tensor = torch.zeros(req_padding, emb_dim)
            res = torch.cat((res, pad_tensor), dim=0)
        except RuntimeError:
            print("res b4 pad: ", res)
            print("padding while pad: ", req_padding)
            print("pad_tensor while pad: ", pad_tensor)
    elif res.shape[0] > sentence_length:
        res = res[:sentence_length, :]

    
    if to_list: return res.tolist()
    else: return res

def get_fasttext_word_vectors(text: str, to_list: bool = False):
    """Generates word vectors in the format of FastText, using torch.vocab.

    Args:
        text (str): Input string.
        to_list (bool, optional): If True, returns a 2d array with the word vectors. Defaults to False.

    Returns:
        list or torch.tensor: A 2D array containing all the word vectors. Defaults to torch.tensor.
    """
    tokenizer = RegexpTokenizer("[\w']+")
    words = tokenizer.tokenize(text)
    vec = FastText(language="en") # 6.6GB
    res = vec.get_vecs_by_tokens(words, lower_case_backup=True)
    if to_list: return res.tolist()
    else: return res

def _test(text, embedding_type: str, to_list: bool = False):
    assert embedding_type == "bert" or embedding_type == "glove" or embedding_type == "fasttext", "Type not defined!"

    if embedding_type == "bert":
        return(get_bert_word_embeddings(text, to_list=to_list))
    elif embedding_type == "glove":
        return(get_glove_word_vectors(text, to_list=to_list))
    elif embedding_type == "fasttext":
        return(get_fasttext_word_vectors(text, to_list=to_list))

def save(text, embedding_type: str, path: str):
    assert embedding_type == "bert" or embedding_type == "glove" or embedding_type == "fasttext", "Type not defined!"
    
    if embedding_type == "bert":
        torch.save(get_bert_word_embeddings(text), path)
    elif embedding_type == "glove":
        torch.save(get_glove_word_vectors(text), path)
    elif embedding_type == "fasttext":
        torch.save(get_fasttext_word_vectors(text), path)

def load(path):
    return torch.load(path)


if __name__ == "__main__":
    # Type
    embedding_type = "bert" 
    # embedding_type = "glove" 
    # embedding_type = "fasttext"

    # Text
    example1 = "It does not do to dwell on dreams and forget to live, remember that. Now, why don’t you put that admirable Cloak back on and get off to bed?"
    example2 = "Just because you’ve got the emotional range of a teaspoon doesn’t mean we all have."
    example3 = "Voldemort himself created his worst enemy, just as tyrants everywhere do! Have you any idea how much tyrants fear the people they oppress? All of them realize that, one day, amongst their many victims, there is sure to be one who rises against them and strikes back!"
    
    # Test
    """ test1 = _test(example1, embedding_type, to_list=True)
    test2 = _test([example1, example2], embedding_type, to_list=True)
    
    print(len(test1))
    print(len(test2))
    print(len(test2[0]))
    """
    
    ex_test = preprocess_text(example2)
    test_tokens = []
    for i in range(20):
        for word in ex_test:
            test_tokens.append(word)

    test_emb = get_glove_word_vectors(test_tokens, 50)
    print(test_emb.shape)

    # Save
    # path = Path(os.path.abspath(__file__)).parents[1] / "features" / "embeddings" / f"example1_{embedding_type}.pt"
    # save(example1, embedding_type, path)

    # Load
    # tensor = load(path).tolist()
    # print(tensor)
