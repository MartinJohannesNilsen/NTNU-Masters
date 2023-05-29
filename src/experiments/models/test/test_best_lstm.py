import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import tune, get_gpu_ids
from ray.air import session, RunConfig
from ray.air.checkpoint import Checkpoint
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.experiment.trial import Trial
from pathlib import Path
import sys
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import h5py
import click
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
from utils.extract_best_model_and_stats import get_best_scoring_config

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

# Hyperparam tuning made from guide by https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html#tune-pytorch-cifar-ref

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

# DATASETS
class TextDatasetH5py(Dataset):
    def __init__(self, emb_path, max_len, pad_pos):
        self.emb_f = h5py.File(emb_path, "r")
        self.max_len = max_len
        self.pad_pos = pad_pos

    def __len__(self):
        return len(self.emb_f["label"])

    def __getitem__(self, i):
        embs = self.emb_f["emb_tensor"][i]
        label = self.emb_f["label"][i]
        length = self.emb_f["length"][i]

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
    
    def get_class_weights(self):
        labels = self.emb_f["label"]

        total_texts = labels.shape[0]
        num_shooter_texts = sum(labels)
        num_non_shooter_texts = total_texts - num_shooter_texts
        non_shooter_wt = total_texts / num_non_shooter_texts
        shooter_wt = total_texts / num_shooter_texts

        print(f"non_shooter: {non_shooter_wt}\nshooter: {shooter_wt}")

        return [non_shooter_wt, shooter_wt]


device = "cuda" if torch.cuda.is_available() else "cpu"


model_to_dim = {
    "glove_50": 50,
    "glove": 300,
    "fasttext": 300,
    "bert": 768
}

torch.manual_seed(0)
def train(emb_dim: int, hidden_size: int, dropout: float, num_layers: int, emb_type: str, pad_pos: str, max_len: int, batch_size: int, lr: float):

    emb_str = f"{emb_type}_{model_to_dim[emb_type]}" if emb_type != "glove_50" else emb_type

    base_path_embs = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings" / "new"
    print(f"base_path: {base_path_embs}")
    train_path_embs = base_path_embs / f"train_sliced_stair_twitter_{emb_str}_{pad_pos}_{max_len}.h5"
    val_path_embs = base_path_embs / f"val_sliced_stair_twitter_{emb_str}_{pad_pos}_{max_len}.h5"
    
    train_set = TextDatasetH5py(train_path_embs, max_len, pad_pos)
    val_set = TextDatasetH5py(val_path_embs,  max_len, pad_pos)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)

    model = LSTMTextClassifier(emb_dim=emb_dim, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)
    print(f"device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    class_wts = train_set.get_class_weights()

    for epoch in range(10):  # loop over the dataset multiple times
        print(f"epoch: {epoch}")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            emb_inputs, labels, start, end = data

            weighting = []
            for l in labels:
                if l == 0:
                    weighting.append(class_wts[0])
                else:
                    weighting.append(class_wts[1])

            emb_inputs, labels, weighting = emb_inputs.to(device), labels.to(float).to(device), torch.tensor(weighting).to(device)
            criterion = nn.BCELoss(weight=weighting)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(emb_inputs, start, end).to(float)
            loss = criterion(outputs.squeeze(dim=1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            
        avg_loss = running_loss / len(train_loader)

        # Validation loss
        val_loss = 0.0
        pred_vlabels = []
        true_vlabels = []
        with torch.no_grad():
            for i, vdata in enumerate(val_loader, 0):
                vemb_inputs, vlabels, vstart, vend = vdata
                vemb_inputs, vlabels = vemb_inputs.to(device), vlabels.to(float).to(device)

                voutputs = model(vemb_inputs, vstart, vend).to(float)

                vweighting = []
                for l in vlabels:
                    if l == 0:
                        vweighting.append(class_wts[0])
                    else:
                        vweighting.append(class_wts[1])
                vweighting = torch.tensor(vweighting).to(device)

                criterion = nn.BCELoss(weight=vweighting)
                vloss = criterion(voutputs.squeeze(dim=1), vlabels)
                val_loss += vloss.item()

                voutputs = voutputs.cpu()
                vlabels = vlabels.cpu()

                [true_vlabels.append(vlabel) for vlabel in vlabels]
                [pred_vlabels.append(1) if pred > 0.5 else pred_vlabels.append(0) for pred in voutputs[0]]

        avg_vloss = val_loss/len(val_loader)    

        metrics = get_metrics(pred_vlabels, true_vlabels)
        metrics["epoch"] = epoch
        metrics["avg_train_loss"] = avg_loss
        metrics["loss"] = avg_vloss
        print(f"metrics:\n{metrics}")

    print("Finished Training")
    return model


def test_model(model, config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    emb_str = f"{config['emb_type']}_{model_to_dim[config['emb_type']]}" if config['emb_type'] != "glove_50" else config['emb_type']
    base_path_embs = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings"
    test_path_embs = base_path_embs / f"shooter_hold_out_{emb_str}_{config['pad_pos']}_{config['max_len']}.h5"

    test_set = TextDatasetH5py(test_path_embs, config["max_len"], config["pad_pos"])
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, pin_memory=True) 

    true_labels = []
    pred_labels = []
    pred_values = []
    with torch.no_grad():
        for data in test_loader:
            emb_inputs, labels, start, end = data
            emb_inputs, labels = emb_inputs.to(device), labels.to(device)
            outputs = model(emb_inputs, start, end).to(float)

            outputs = outputs.cpu()
            labels = labels.cpu()

            [true_labels.append(label.item()) for label in labels]
            [pred_labels.append(1) if pred > 0.5 else pred_labels.append(0) for pred in outputs]
            [pred_values.append(output.item()) for output in outputs]


    metrics = get_metrics(pred_labels, true_labels)
    print(f"Best results with config:\n{config}")
    print(f"Got metrics: {metrics}")
    return pred_values, pred_labels, true_labels


if __name__ == "__main__":
    best_emb_type, best_config, best_score = get_best_scoring_config("lstm", "f2_score")
    
    print(f"Best emb type: {best_emb_type}")
    print(f"Best config:\n{best_config}")
    print(f"Best score:\n{best_score}")

    best_model = train(
        emb_dim=best_config["emb_dim"],
        hidden_size=best_config["hidden_size"],
        dropout=best_config["dropout"],
        num_layers=best_config["num_layers"],
        emb_type=best_config["emb_type"],
        pad_pos=best_config["pad_pos"],
        max_len=best_config["max_len"],
        batch_size=best_config["batch_size"],
        lr=best_config["lr"]
    )

    pred_values, pred_labels, true_labels = test_model(best_model, best_config)

    lstm_out_path = Path(os.path.abspath(__file__)).parents[2] / "pred_values" / "shooter_hold_out" / "lstm_preds.csv"
    lstm_df = pd.DataFrame({"pred_val": pred_values, "pred_label": pred_labels, "label": true_labels})
    lstm_df.to_csv(lstm_out_path, index=False)
