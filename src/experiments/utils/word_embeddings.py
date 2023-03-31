# Imports
import os
from pathlib import Path
import numpy as np
import torch
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from torchtext.vocab import FastText, GloVe
from transformers import AutoTokenizer, AutoModel

import re
from typing import List
from transformers import pipeline

def get_bert_word_embeddings(input, pretrained_name = "bert-base-uncased", to_list: bool = True, max_length: int = None):
    """A method for generating BERT word embeddings using a pre-trained tokenizer. Returns result as either torch tensors (default) or regular lists. 

    Args:
        input (str or List[str]): Input string or list of input strings.
        chunk_size (int, optional): The size of chunks. Padding will be applied. Defaults to 512.
        pretrained_tokenizer (str, optional): Which pre-trained tokenizer to use. Defaults to 'bert-base-uncased'.
        do_lower_case (bool, optional): If True, lowercasing will be applied. Defaults to False.

    Returns:
        list or torch.tensor: A 2d array of chunked word embeddings. Defaults to torch.tensor.
    """

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = AutoModel.from_pretrained(pretrained_name)

    extract_features = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
    if max_length:
        out = extract_features(input, padding='max_length', truncation=True, max_length=max_length)
    else:    
        out = extract_features(input)

    return out

       
def _tokenize_with_preprocessing(text: str, remove_url: bool = True):
    """For the methods needing tokenized input. The method perform certain steps of text cleaning:
        - Stopword removal
        - Url replacement (token or removal)

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
            word = re.sub(r'([A-Za-z])\1{2,}', r'\1', word) # Character normalization, prevent words with letters repeated more than twice

            if word != "":
                cleaned_words.append(word)

    #words = [re.sub(r'http+|www+', url_replacement, word).lower() for word in words if word not in stopwords.words("english")] 

    #return [word for word in words if word != ""]
    return cleaned_words

    
def get_glove_word_vectors(input: str or List[str], min_length: int = None, emb_dim: 50 or 300 = 50):
    """Generates word vectors in the format of GloVe, using torch.vocab.

    Args:
        text (str): Input string.
        size_small (bool, optional): If False, use the 2.18GB pre-trained model instead of the 862MB one. Defaults to True.
        min_length (int, optional): If defined, pads the list of tokens to desired length if smaller than value. Defaults to None.
        emb_dim (int, optional): If defined, we utilize the GloVe model of defined size, instead of 300. Defaults to 50.

    Returns:
        list or torch.tensor: If input is string, returns a list (to_list=True) or tensor (to_list=False) of dimensions (n_tokens, emb_dim). 
                              If min_length is defined and higher than input length, the n_tokens will be padded to that length. 
                              If input is a list of strings, returns a list of either list or tensors based on to_list, with dimensions (n_inputs, n_tokens, emb_dim).
    """
    # vec = GloVe(name='6B', dim=50) # 862MB
    # vec = GloVe(name='840B', dim=300) # 2.18GB

    def _extract_embeddings(input):
        tokenized_input = _tokenize_with_preprocessing(input)
        try:
            glove_vec = GloVe(name='6B', dim=50) if emb_dim == 50 else GloVe(name='840B', dim=300)
            res = glove_vec.get_vecs_by_tokens(tokenized_input, lower_case_backup=True)
        except RuntimeError:
            print("Input words: ", tokenized_input)
        
        if min_length:
            if res.shape[0] < min_length:
                try:
                    req_padding = min_length - res.shape[0]
                    pad_tensor = torch.zeros(req_padding, emb_dim)
                    res = torch.cat((res, pad_tensor), dim=0)
                except RuntimeError:
                    print("res b4 pad: ", res)
                    print("padding while pad: ", req_padding)
                    print("pad_tensor while pad: ", pad_tensor)
        
        return res

    if isinstance(input, str):
        return _extract_embeddings(input)
    else:
        return [_extract_embeddings(text) for text in input]
    

def get_fasttext_word_vectors(input: str or List[str]):
    """Generates word vectors in the format of FastText, using torch.vocab.

    Args:
        text (str): Input string.

    Returns:
        list or torch.tensor: If input is string, returns a list (to_list=True) or tensor (to_list=False) of dimensions (n_tokens, 300). 
                              If input is a list of strings, returns a list of either list or tensors based on to_list, with dimensions (n_inputs, n_tokens, 300).
    """
    def _extract_embeddings(input):
        tokenized_input = _tokenize_with_preprocessing(input)
        vec = FastText(language="en") # 6.6GB
        res = vec.get_vecs_by_tokens(tokenized_input, lower_case_backup=True)
        return res

    if isinstance(input, str):
        return _extract_embeddings(input)
    else:
        return [_extract_embeddings(text) for text in input]
    

    if to_list: return res.tolist()
    else: return res

def _test(text, embedding_type: str, max_length: int = None):
    assert embedding_type == "bert" or embedding_type == "glove" or embedding_type == "fasttext", "Type not defined!"

    if embedding_type == "bert":
        return(get_bert_word_embeddings(text, max_length=max_length))
    elif embedding_type == "glove":
        return(get_glove_word_vectors(text, min_length=max_length))
    elif embedding_type == "fasttext":
        return(get_fasttext_word_vectors(text))

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
    # embedding_type = "bert" 
    # embedding_type = "glove"
    # embedding_type = "fasttext"

    # Text
    example1 = "It does not do to dwell on dreams and forget to live, remember that. Now, why don’t you put that admirable Cloak back on and get off to bed?"
    example2 = "Just because you’ve got the emotional range of a teaspoon doesn’t mean we all have."
    # example3 = "Voldemort himself created his worst enemy, just as tyrants everywhere do! Have you any idea how much tyrants fear the people they oppress? All of them realize that, one day, amongst their many victims, there is sure to be one who rises against them and strikes back!"
    
    # Test
    # test1 = _test(example1, embedding_type, to_list=False, max_length=512)
    # test1 = _test(example1, embedding_type, to_list=False)
    # test1 = _test([example1, example2], embedding_type, to_list=False, max_length=512)
    bert = _test([example1, example2], "bert")
    glove = _test([example1, example2], "glove")
    fasttext = _test([example1, example2], "fasttext")

    print("start")
    
















    
    # Save
    # path = Path(os.path.abspath(__file__)).parents[1] / "features" / "embeddings" / f"example1_{embedding_type}.pt"
    # save(example1, embedding_type, path)

    # Load
    # tensor = load(path).tolist()
    # print(tensor)
