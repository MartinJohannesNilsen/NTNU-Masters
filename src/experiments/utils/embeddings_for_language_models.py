# Imports
import torch
from transformers import BertTokenizer


def get_bert_word_embeddings_lm(text: str, chunk_size: int = 512, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False), truncate: bool = False):
    """A method for generating BERT word embeddings using a pre-trained tokenizer. Returns result as either torch tensors (default) or regular lists. NB! This is only outputing the dictionary for a language model with input ids, not the actual embeddings.

    Args:
        text (str): Input string.
        chunk_size (int, optional): The size of chunks. Padding will be applied. Defaults to 512.
        pretrained_tokenizer (str, optional): Which pre-trained tokenizer to use. Defaults to 'bert-base-uncased'.
        do_lower_case (bool, optional): If True, lowercasing will be applied. Defaults to False.
        to_list (bool, optional): If True, returns a 2d array with the chunked lists as input_ids in the final dictionary. Defaults to False.

    Returns:
        dict: A dictionary of inout ids and attention masks.
    """
    
    # Tokenize and return tensors
    if truncate:
        tokens = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors="pt", truncation=truncate, max_length = chunk_size)
    else:   
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

    # Create return dictionary
    input_dict = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.int()
    }

    return input_dict
