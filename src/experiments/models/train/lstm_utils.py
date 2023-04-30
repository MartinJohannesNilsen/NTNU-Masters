import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import h5py as h5py

# MODEL ARCHITECTURES

class LSTMTextClassifier(nn.Module):
    def __init__(self, emb_wts: None, emb_dim: int = 300, dropout: int = 0.5, hidden_size: int = 128, num_layers: int = 2):
        super(LSTMTextClassifier, self).__init__()

        self.emb_layer = nn.Embedding.from_pretrained(torch.from_numpy(emb_wts).float()) if emb_wts else None
        self.embedding.weight.requires_grad = False
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.sig = nn.Sigmoid()


    def forward(self, x, length):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): A tensor of token ids with shape (batch_size, max_sent_length)
            length: the length of the sequence before padding
        """

        embs = self.embedding(x) if self.emb_layer else x

        packed_input = pack_padded_sequence(embs, length, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_input)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        out_forward = out[range(len(out)), length - 1, :self.hidden_size]
        out_backwards = out[:, 0, self.hidden_size:]

        out_reduced = torch.cat((out_forward, out_backwards), 1) # Concat for fc layer and final pred
        out_dropped = self.dropout(out_reduced) # Dropout layer
        out = self.fc(out_dropped)
        out = self.sig(out)

        return out
    


# DATASETS
class TextDataset(Dataset):
    def __init__(self, path):
        self.f = h5py.File(path, "r")

    def __len__(self):
        return len(self.f["label"])

    def __getitem__(self, idx):
        embs = self.f["emb_tensor"][idx]
        label = self.f["label"][idx]
        #seq_len = self.f["seq_len"][idx]

        return embs, label
    
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
        tokens, length = row[1]
        label = row[3]

        return tokens, label, length
    
    def get_class_weights(self):
        total_texts = self.__len__()
        num_non_shooter_texts, num_shooter_texts = self.df["label"].value_counts()
        print(f"Value counts:\n{self.df['label'].value_counts()}")

        non_shooter_wt = total_texts / num_non_shooter_texts
        shooter_wt = total_texts / num_shooter_texts
        print(f"non_shooter: {non_shooter_wt}\nshooter: {shooter_wt}")

        return [non_shooter_wt, shooter_wt]