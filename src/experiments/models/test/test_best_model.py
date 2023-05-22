import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import h5py
import click
from sklearn.metrics import confusion_matrix, roc_auc_score
import math
import pandas as pd

def get_metrics(predictions, labels) -> dict:
    """Get a selection of performance metrics.

    Args:
        predictions (List[float]): List of predictions.
        labels (List[float]): List of labels.

    Returns:
        dict: Dictionary of performance metrics.
    """
    
    # Gather performance metrics
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    specificity = (tn) / (tn + fp)

    fscore = 2 * (precision * recall) / (precision + recall)
    beta = 0.5
    f05score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    beta = 2
    f2score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    
    try:
        roc_auc = roc_auc_score(labels, predictions)
    except:
        roc_auc = None
    
    # Create dictionary of metrics
    metrics = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": fscore,
        "roc_auc": roc_auc,
        "f2_score": f2score,
        "f_05_score": f05score
    }

    return metrics

class LSTMTextClassifier(nn.Module):
    def __init__(self, emb_wts = None, emb_dim: int = 300, dropout: int = 0.5, hidden_size: int = 128, num_layers: int = 2):
        super(LSTMTextClassifier, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(emb_wts).float()) if emb_wts else None
        if emb_wts: self.embedding.weight.requires_grad = False
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.sig = nn.Sigmoid()


    def forward(self, emb_tensors, idx_first_hidden, idx_last_hidden):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): A tensor of token ids with shape (batch_size, max_sent_length)
            length: the length of the sequence before padding
        """
        out, _ = self.lstm(emb_tensors)
        out_forward = out[range(len(out)), idx_last_hidden, :self.hidden_size] # Get output of last valid LSTM element, not padding
        out_backwards = out[range(len(out)), idx_first_hidden, self.hidden_size:] # Output of first node thingy thangy
        out_reduced = torch.cat((out_forward, out_backwards), 1) # Concat for fc layer and final pred
        out_dropped = self.dropout(out_reduced) # Dropout layer
        out = self.fc(out_dropped)
        out = self.sig(out)

        return out


class CNNTextClassifier(nn.Module):
    def __init__(self, emb_dim: int = 300, filter_sizes = [3,4,5], num_filters = [100,100,100], dropout: int = 0.5):
        super(CNNTextClassifier, self).__init__()

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.fc = nn.Linear(np.sum(num_filters), 1)
        self.dropout = nn.Dropout(p=dropout)
        self.sig = nn.Sigmoid() # Sigmoid to squeeze final vals between 0 and 1 to accomodate for binary class prob

    def forward(self, x):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Input shape: (b, max_len, embed_dim)
        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Output shape: (b, n_classes) ---> (b, 1)
        out = self.fc(self.dropout(x_fc))

        # Squeeze between values 0 and 1 (non-shooter and shooter)
        out = self.sig(out)

        return out

class TextDatasetH5py(Dataset):
    def __init__(self, path, pad_pos, max_len=256):
        self.f = h5py.File(path, "r")
        self.pad_pos = pad_pos
        self.max_len = max_len

    def __len__(self):
        return len(self.f["label"])

    def __getitem__(self, idx):
        embs = self.f["emb_tensor"][idx]
        label = self.f["label"][idx]
        length = self.f["length"][idx]

        start = 0
        end = start + (length - 1)
        if self.pad_pos == "head":
            start = -length
            end = -1
        elif self.pad_pos == "split":
            req_padding = self.max_len - length
            start = math.floor(req_padding/2)
            end = start + length - 1

        #print(f"start: {start}, end: {end}, length: {length}")

        return embs, label, start, end


def test_model(model, model_type, test_path, pad_pos, device):

    ds = TextDatasetH5py(test_path, pad_pos)
    test_loader = DataLoader(ds, batch_size=1, shuffle=False)

    preds = []
    pred_vals = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, start, end = data
            inputs, labels = inputs.to(device), labels.to(float).to(device)

            outputs = model(inputs, start, end).to(float) if model_type == "lstm" else model(inputs).to(float)
            outputs = outputs.cpu()
            labels = labels.cpu()

            [true_labels.append(label.item()) for label in labels]
            [preds.append(1) if pred > 0.5 else preds.append(0) for pred in outputs]
            [pred_vals.append(output.item()) for output in outputs]

    print(get_metrics(true_labels, preds))
    return pred_vals, preds, true_labels


if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    emb_type = "bert"
    emb_dim = 768

    emb_str = f"{emb_type}_{emb_dim}"
    base_path = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings" / "new"
    lstm_test_path = base_path / f"test_sliced_stair_twitter_{emb_str}_head_256.h5"
    cnn_test_path = base_path / f"test_sliced_stair_twitter_{emb_str}_split_256.h5"

    gs_results_path = Path(os.path.abspath(__file__)).parents[2] / "models" / "train" / "gs_results"
    lstm_path = gs_results_path / "lstm_gpu" / "model_bert_768_256" / "best_lstm_model" / "checkpoint_000009" / "checkpoint.pt"
    cnn_path = gs_results_path / "cnn" / "model_bert_768_256" / "best_cnn_model" / "checkpoint_000008" / "checkpoint.pt"
    
    lstm_model = LSTMTextClassifier(emb_dim=768, dropout=0.6, hidden_size=64, num_layers=2).to(device)
    lstm_model.load_state_dict(torch.load(lstm_path)[0])
    lstm_out_path = Path(os.path.abspath(__file__)).parents[2] / "pred_values" / "lstm_preds.csv"
    lstm_vals, lstm_preds, lstm_labels = test_model(lstm_model, "lstm", lstm_test_path, "split", device)
    lstm_df = pd.DataFrame({"pred_val": lstm_vals, "pred_label": lstm_preds, "label": lstm_labels})
    lstm_df.to_csv(lstm_out_path, index=False)

    cnn_model = CNNTextClassifier(emb_dim=768, dropout=0.3).to(device)
    cnn_model.load_state_dict(torch.load(cnn_path)[0])
    cnn_out_path = Path(os.path.abspath(__file__)).parents[2] / "pred_values" / "cnn_preds.csv"
    cnn_vals, cnn_preds, cnn_labels = test_model(cnn_model, "cnn", cnn_test_path, "head", device)
    cnn_df = pd.DataFrame({"pred_val": cnn_vals, "cnn_label": cnn_preds, "label": cnn_labels})
    cnn_df.to_csv(cnn_out_path, index=False)

