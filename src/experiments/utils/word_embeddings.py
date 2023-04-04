# Imports
from bs4 import BeautifulSoup
import torch
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from torchtext.vocab import FastText, GloVe
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
import re
from typing import List
from transformers import pipeline

def _apply_fixed_sentence_length(embedding: torch.tensor, sentence_length: int, emb_dim: int) -> torch.tensor:
    """Pad if embedding is smaller then sentence length, and truncate if longer.

    Args:
        embedding (torch.tensor): Embedding from the torch.vocab or transformers pipeline functions.
        sentence_length (int): Fixed sentence length.
        emb_dim (int): The last dimensionality value.

    Returns:
        torch.tensor: New tensor with fixed sentence length.
    """
    
    if embedding.shape[0] < sentence_length:
        try:
            req_padding = sentence_length - embedding.shape[0]
            pad_tensor = torch.zeros(req_padding, emb_dim)
            embedding = torch.cat((embedding, pad_tensor), dim=0)
        except RuntimeError:
            print("embedding b4 pad: ", embedding)
            print("padding while pad: ", req_padding)
            print("pad_tensor while pad: ", pad_tensor)
    elif embedding.shape[0] > sentence_length:
        embedding = embedding[:sentence_length, :]
    return embedding


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
            word = re.sub(r'([A-Za-z])\1{2,}', r'\1', word) # Character normalization, prevent words with letters repeated more than twice

            if word != "":
                cleaned_words.append(word)

    return cleaned_words


def get_bert_word_embeddings(input: str or List[str], pretrained_name = "bert-base-uncased", sentence_length: int = None):
    """Generates BERT word embeddings, using the Transformers pipeline. NB! The batch dimension is squeezed out as default.

    Args:
        input (str or List[str]): Input string or list of input strings.
        pretrained_name (str, optional): Which pre-trained model to use. Defaults to 'bert-base-uncased'.
        chunk_size (int, optional): The size of chunks. Padding will be applied. Defaults to 512.

    Returns:
        torch.tensor or List[torch.tensor]: If input is string, returns a tensor of dimensions (n_tokens, emb_dim=768). 
                                            If input is a list of strings, returns a list of tensors with dimensions (n_tokens, emb_dim=768).
                                            Note that n_tokens will be equal to sentence_length if defined.

    """

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
        return _apply_fixed_sentence_length(torch.tensor(out).squeeze(), sentence_length=sentence_length, emb_dim=emb_dim) if sentence_length else torch.tensor(out).squeeze()
    else:
        return [_apply_fixed_sentence_length(torch.tensor(e).squeeze(), sentence_length=sentence_length, emb_dim=emb_dim) for e in out] if sentence_length else [torch.tensor(e).squeeze() for e in out]


    
def get_glove_word_vectors(input: str or List[str], sentence_length: int = None, size_small: bool = True):
    """Generates word vectors in the format of GloVe, using torch.vocab.

    Args:
        input (str or List[str]): Input string or list of input strings.
        size_small (bool, optional): If False, use the 2.18GB pre-trained model instead of the 862MB one. Defaults to True. Emb_dim is 50 for small, and 300 for large.
        sentence_length (int, optional): If defined, pads the list of tokens to desired length. Truncates if longer. Defaults to None.

    Returns:
        torch.tensor or List[torch.tensor]: If input is string, returns a tensor of dimensions (n_tokens, emb_dim=50 or 300). 
                                            If input is a list of strings, returns a list of tensors with dimensions (n_tokens, emb_dim=50 or 300).
                                            Note that n_tokens will be equal to sentence_length if defined.
    """
    def _extract_embeddings(input):
        tokenized_input = _tokenize_with_preprocessing(input)

        if len(tokenized_input) == 0:
            return

        emb_dim = 50 if size_small else 300
        glove_vec = GloVe(name='6B', dim=50) if emb_dim == 50 else GloVe(name='840B', dim=300)
        res = glove_vec.get_vecs_by_tokens(tokenized_input, lower_case_backup=True)
        
        res = _apply_fixed_sentence_length(res, sentence_length=sentence_length, emb_dim=emb_dim)
        
        return res

    if isinstance(input, str):
        return _extract_embeddings(input)
    else:
        return [_extract_embeddings(text) for text in input]
    

def get_fasttext_word_vectors(input: str or List[str], sentence_length: int = None):
    """Generates word vectors in the format of FastText, using torch.vocab.

    Args:
        input (str or List[str]): Input string or list of input strings.
        sentence_length (int, optional): If defined, pads the list of tokens to desired length. Truncates if longer. Defaults to None.

    Returns:
        torch.tensor or List[torch.tensor]: If input is string, returns a tensor with dimensions (n_tokens, emd_dim=300). 
                                            If input is a list of strings, returns a list of tensors with dimensions (n_tokens, emd_dim=300).
    """

    def _extract_embeddings(input):
        tokenized_input = _tokenize_with_preprocessing(input)

        if len(tokenized_input) == 0:
            return
        
        vec = FastText(language="en") # 6.6GB
        res = vec.get_vecs_by_tokens(tokenized_input, lower_case_backup=True)
        
        emb_dim = 300
        res = _apply_fixed_sentence_length(res, sentence_length=sentence_length, emb_dim=emb_dim)
        
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
    bert = get_bert_word_embeddings(example1, sentence_length=5) # Tensor of dim (5, 768)
    # glove = get_glove_word_vectors(example1, sentence_length=5) # Tensor of dim (5, 50)
    # fasttext = get_fasttext_word_vectors(example1, sentence_length=5) # Tensor of dim (5, 300)

    print(bert.shape)