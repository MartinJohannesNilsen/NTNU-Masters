# Imports
import torch
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from torchtext.vocab import FastText, GloVe
from transformers import BertTokenizer
from bs4 import BeautifulSoup
import string
import re
import traceback
from typing import List
import torch.nn.functional as F


def get_bert_word_embeddings(text: str, chunk_size: int = 512, pretrained_tokenizer: str = 'bert-base-uncased', do_lower_case: bool = False, to_list: bool = False):
    """A method for generating BERT word embeddings using a pre-trained tokenizer. Returns result as either torch tensors (default) or regular lists.

    Args:
        text (str): Input string.
        chunk_size (int, optional): The size of chunks. Padding will be applied. Defaults to 512.
        pretrained_tokenizer (str, optional): Which pre-trained tokenizer to use. Defaults to 'bert-base-uncased'.
        do_lower_case (bool, optional): If True, lowercasing will be applied. Defaults to False.
        to_list (bool, optional): If True, returns a 2d array with the chunked lists. Defaults to False.

    Returns:
        list or torch.tensor: A 2d array of chunked word embeddings. Defaults to torch.tensor.
    """

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer, do_lower_case=do_lower_case)
    
    # Tokenize and return tensors
    tokens = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors="pt")
    
    # Split into chunks of 510 tokens, we also convert to list (default is tuple which is immutable)
    input_id_chunks = list(tokens['input_ids'][0].split(chunk_size - 2))
    mask_chunks = list(tokens['attention_mask'][0].split(chunk_size - 2))

    # Loop through each chunk
    for i in range(len(input_id_chunks)):
        # Add CLS and SEP tokens to input IDs
        input_id_chunks[i] = torch.cat([torch.tensor([101]), input_id_chunks[i], torch.tensor([102])])
        # Add attention tokens to attention mask
        mask_chunks[i] = torch.cat([torch.tensor([1]), mask_chunks[i], torch.tensor([1])])
        # Get required padding length
        pad_len = chunk_size - input_id_chunks[i].shape[0]
        # Check if tensor length satisfies required chunk size
        if pad_len > 0:
            # If padding length is more than 0, we must add padding
            input_id_chunks[i] = torch.cat([input_id_chunks[i], torch.Tensor([0] * pad_len)])
            mask_chunks[i] = torch.cat([mask_chunks[i], torch.Tensor([0] * pad_len)])

    # Create stacks
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)

    if to_list:
        return input_ids.tolist()

    # Create return dictionary
    input_dict = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.int()
    }

    return input_dict


def preprocess_text(text: str, full_clean_url: bool = True):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    tokenizer = RegexpTokenizer("[\w']+")
    words = tokenizer.tokenize(text)

    url_replacement = "" if full_clean_url else "URLHYPERLINK"

    words = [re.sub(r'http+|www+', url_replacement, word).lower() for word in words if word not in stopwords.words("english")] 

    return [word for word in words if word != ""]

    
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
        req_padding = sentence_length - res.shape[0]
        pad_tensor = torch.zeros(req_padding, emb_dim)
        res = torch.cat((res, pad_tensor), dim=0)
    
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


if __name__ == "__main__":
    example1 = "It does not do to dwell on dreams and forget to live, remember that. Now, why don’t you put that admirable Cloak back on and get off to bed?"
    example2 = "Just because you’ve got the emotional range of a teaspoon doesn’t mean we all have."
    example3 = "Voldemort himself created his worst enemy, just as tyrants everywhere do! Have you any idea how much tyrants fear the people they oppress? All of them realize that, one day, amongst their many victims, there is sure to be one who rises against them and strikes back!"

    # print(get_bert_word_embeddings(example1, to_list=True))
    # print(get_glove_word_vectors(example1))
    print(get_fasttext_word_vectors(example1, to_list=True))
