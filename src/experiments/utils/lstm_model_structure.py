import torch
from torch import nn

class LSTMTextClassifier(nn.Module):
    def __init__(self, emb_wts = None, emb_dim: int = 300, dropout: int = 0.5, hidden_size: int = 128, num_layers: int = 2, liwc_size: int = None):
        super(LSTMTextClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear((2*hidden_size) + liwc_size, 1) if liwc_size else nn.Linear(2*hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.sig = nn.Sigmoid()


    def forward(self, emb_tensors, idx_first_hidden, idx_last_hidden, liwc_tensor = None):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): A tensor of token ids with shape (batch_size, max_sent_length)
            length: the length of the sequence before padding
        """
        out, _ = self.lstm(emb_tensors)

        out_forward = out[range(len(out)), idx_last_hidden, :self.hidden_size] # Get output of last valid LSTM element, not padding
        out_backwards = out[range(len(out)), idx_first_hidden, self.hidden_size:] # Output of first node thingy thangy

        out_reduced = torch.cat((out_forward, out_backwards, liwc_tensor), 1) if liwc_tensor != None else torch.cat((out_forward, out_backwards), 1) # Concat for fc layer and final pred
        out_dropped = self.dropout(out_reduced) # Dropout layer
        out = self.fc(out_dropped)
        out = self.sig(out)

        return out