from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import h5py
import sys
import os
from pathlib import Path
import numpy as np

sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))

base_path = Path(os.path.abspath(__file__)).parents[1] / "features" / "embeddings"
train_path = base_path / "train_sliced_stair_twitter_glove_head.h5"

f = h5py.File(train_path)

print(f["emb_tensor"][0][-1].shape)
print(f["emb_tensor"][0][-1])
print(type(f["emb_tensor"][0][-1]))
print(f"any non zero? {np.any(f['emb_tensor'][0][-1])}")
print(f"sum of arr at -1? {np.sum(f['emb_tensor'][0][-1])}")

""" ones = np.ones(10)
zeros = np.zeros(10)
whole = np.concatenate([ones,zeros], axis=0)
whole2 = np.copy(whole)
t1 = torch.tensor([whole, whole2])
print(t1.shape[-1])
print(t1)

[print(t1[i]) for i in range(t1.shape[0])] """

def get_seq_len(seq):
    """
    Due to the way embeddings were stored at the beginning of the project, extracting lengths of the individual sequences was deemed necessary
    """
    whole_seq_len = len(seq)
    print(whole_seq_len)
    print(seq[7])

    i = 1
    while i < whole_seq_len:
        if np.any(seq[-i]):
            break

        i += 1
        
    return whole_seq_len - (i-1)

"""
while not np.any(seq[-i+1]):
    print(i)
    i += 1

return whole_seq_len - (i-1) """

print(get_seq_len(f["emb_tensor"][0]))

""" tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
extract_features = pipeline('feature-extraction', model=model, tokenizer=tokenizer, padding="max_length", max_length=10, truncation=True)
print(torch.tensor(extract_features(test_str)).squeeze()[1]) """
