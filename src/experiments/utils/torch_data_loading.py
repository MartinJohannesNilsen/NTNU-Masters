import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import h5py as h5py

# DATASETS
class TextDatasetH5py(Dataset):
    def __init__(self, path):
        self.f = h5py.File(path, "r")

    def __len__(self):
        return len(self.f["label"])

    def __getitem__(self, idx):
        embs = self.f["emb_tensor"][idx]
        label = self.f["label"][idx]
        seq_len = self.f["seq_len"][idx]

        return embs, label, seq_len
    
    def get_class_weights(self):
        labels = self.f["label"]

        total_texts = labels.shape[0]
        num_shooter_texts = sum(labels)
        num_non_shooter_texts = total_texts - num_shooter_texts
        non_shooter_wt = total_texts / num_non_shooter_texts
        shooter_wt = total_texts / num_shooter_texts

        print(f"non_shooter: {non_shooter_wt}\nshooter: {shooter_wt}")

        return [non_shooter_wt, shooter_wt]
    

class TextDatasetNoEmb(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].values
        tokens = row[1]
        label = row[3]

        return tokens, label
    
    def get_class_weights(self):
        total_texts = self.__len__()
        num_non_shooter_texts, num_shooter_texts = self.df["label"].value_counts()
        print(f"Value counts:\n{self.df['label'].value_counts()}")

        non_shooter_wt = total_texts / num_non_shooter_texts
        shooter_wt = total_texts / num_shooter_texts
        print(f"non_shooter: {non_shooter_wt}\nshooter: {shooter_wt}")

        return [non_shooter_wt, shooter_wt]



def get_data_and_wts(base_path: str, emb_type: str, emb_dim: int, max_len: int = 256, batch_size: int = 32, stair_twitter: bool = False, pad_pos: str = "tail"):
    stair_twitter_str = "sliced_stair_twitter" if stair_twitter else "no_stair_twitter"

    train_path = base_path / f"train_{stair_twitter_str}_{emb_type}_{emb_dim}_{pad_pos}_{max_len}.h5"
    val_path = base_path / f"val_{stair_twitter_str}_{emb_type}_{emb_dim}_{pad_pos}_{max_len}.h5"

    print("Creating datasets...")
    train_set = TextDatasetH5py(train_path)
    val_set = TextDatasetH5py(val_path)

    print("Constructing dataloaders...")

    # Load dataset
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, pin_memory=True)
    class_wts = train_set.get_class_weights() # Make class wts proportional to proportion of class occurences

    return train_loader, val_loader, class_wts



