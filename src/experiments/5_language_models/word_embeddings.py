# Imports
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer


def get_word_embeddings(text: str, chunk_size: int = 512, pretrained_tokenizer: str = 'bert-base-uncased', do_lower_case: bool = False):

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

    # Create return dictionary
    input_dict = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.int()
    }

    return input_dict