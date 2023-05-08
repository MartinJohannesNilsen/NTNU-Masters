import sys 
import os
from pathlib import Path
import torch
import sys
import csv
import pandas as pd
import time
# sys.exit(1)
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
# src/experiments/utils
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, pack_sequence
from datetime import datetime
import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"
import h5py
import numpy as np
from experiments.utils.metrics import get_metrics
from tabulate import tabulate
from experiments.utils.word_embeddings import get_padded_ids, create_vocab_w_idx, get_emb_matrix
from csv import QUOTE_NONE

print(f"device: {device}")


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

def get_seq_len(seq):
    """
    Due to the way embeddings were stored at the beginning of the project, extracting lengths of the individual sequences was deemed necessary
    """

    seq_len = []

    #print(f"seq shape: {seq.shape}")
    if len(seq.shape) == 3:
        #print(f"len whole seq: {len(seq[0])}")
        for s in seq:
            #print(s)
            i = 1
            while i < len(s):
                if torch.any(s[-i]):
                    break

                i += 1
            
            seq_len.append(len(s) - (i-1))
    
    else:
        #print(f"len one seq: {len(seq[0])}")
        i = 1
        while i < len(seq):
            if np.any(seq[-i]):
                break

            i += 1
        
        seq_len.append(len(s) - (i-1))
        
    #print(f"seq_lens: {seq_len}")

    return seq_len


class LSTMTextClassifier(nn.Module):
    def __init__(self, emb_dim: int = 300, dropout: int = 0.5, hidden_size: int = 128, num_layers: int = 2):
        super(LSTMTextClassifier, self).__init__()

        #self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs).float())
        #self.embedding.weight.requires_grad = False
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.sig = nn.Sigmoid()

    def forward(self, x, seq_len):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): A tensor of token ids with shape (batch_size, max_sent_length)
            length: the length of the sequence before padding
        """

        #embs = self.embedding(x)
        #print(f"embs: {embs.shape}")
        #print(f"input shape: {x.shape}")

        #print(f"seq len: {seq_len}")

        packed_input = pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_input)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        out_forward = out[range(len(out)), [(s-1) for s in seq_len], :self.hidden_size]
        out_backwards = out[:, 0, self.hidden_size:]
        print(out_forward)

        out_reduced = torch.cat((out_forward, out_backwards), 1) # Concat for fc layer and final pred
        out_dropped = self.dropout(out_reduced) # Dropout layer
        out = self.fc(out_dropped)
        out = self.sig(out)

        return out


def check_mem_usage():
    print("torch.cuda.memory_allocated: %fMB"%(torch.cuda.memory_allocated(0)/1024/1024))
    print("torch.cuda.memory_reserved: %fMB"%(torch.cuda.memory_reserved(0)/1024/1024))
    print("torch.cuda.max_memory_reserved: %fMB"%(torch.cuda.max_memory_reserved(0)/1024/1024))


def train(emb_type: str, emb_dim: int, pad_pos: str = "tail", num_epochs: int = 10, max_len: int = 256, batch_size: int = 100):
    # 222

    # Read data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0) 

    base_path = Path(os.path.abspath(__file__)).parents[3] / "experiments" / "features" / "embeddings"

    print(f"Fetching data from {base_path}")

    sent_len_str = "" if max_len == 512 else f"_{max_len}"
    emb_str = "" if emb_dim == 300 else f"_{emb_dim}"

    print("Fetching...")

    train_path = base_path / f"train_sliced_stair_twitter_{emb_type}{emb_str}_{pad_pos}{sent_len_str}.h5"
    val_path = base_path / f"test_sliced_stair_twitter_{emb_type}{emb_str}_{pad_pos}{sent_len_str}.h5"


    print("Constructing model...")
    # Create model

    print("mem usage before constructing model")
    check_mem_usage()

    model = LSTMTextClassifier(emb_dim=emb_dim).to(device)

    print("mem usage after creating model")
    check_mem_usage()

    print("Creating datasets...")

    # Creating datasets for use with dataloaders
    train_set = TextDataset(train_path)
    val_set = TextDataset(val_path)

    print("Constructing dataloaders...")

    # Load dataset
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)

    # Create loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Find class wts...")
    class_wts = train_set.get_class_weights() # Make class wts proportional to proportion of class occurences


    def run_epoch():
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(train_loader):
            print(f"batch: {i}")
            t1 = time.time()

            inputs, labels = data
            labels = labels.to(torch.float32).to(device)
            inputs = torch.from_numpy(np.array(inputs)).to(device)
            seq_lens = get_seq_len(inputs)
            print(f"time taken for getting data: {time.time()-t1}")


            optimizer.zero_grad()
            outputs = model(inputs, seq_lens)

            weighting = []
            for l in labels:
                if l == 0:
                    weighting.append(class_wts[0])
                else:
                    weighting.append(class_wts[1])

            weighting = torch.tensor(weighting).to(device)
            loss_fn = nn.BCELoss(weight=weighting)
            loss = loss_fn(outputs.squeeze(dim=1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        return last_loss / len(train_loader)


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0
    EPOCHS = num_epochs
    best_vloss = 1_000_000.

    
    """ wandb.init(
        # set the wandb project where this run will be logged
        project="cnn-glove-features-predict-shooters",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "school-shooters-vs-non-school-shooters",
        "epochs": 10,
        }
    ) """
   

    print("Start training...")

    metrics = {}
    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch}:')

        model.train(True)
        avg_loss = run_epoch()

        print("validating")
        model.train(False)

        pred_vlabels = []
        true_vlabels = []
        running_vloss = 0.0
        with torch.no_grad():
            for vdata in val_loader:
                vinputs, vlabels = vdata
                vinputs = torch.from_numpy(np.array(vinputs)).to(device)

                vlengths = get_seq_len(vinputs)
                v_out = model(vinputs, vlengths)

                [true_vlabels.append(vlabel) for vlabel in vlabels]
                [pred_vlabels.append(1) if pred > 0.5 else pred_vlabels.append(0) for pred in v_out[0]]
                
                weighting = [class_wts[l] for l in vlabels]
                weighting = torch.tensor(weighting).to(device)
                vlabels = vlabels.to(torch.float32).to(device)

                loss_fn = nn.BCELoss(weight=weighting)
                vloss = loss_fn(v_out.squeeze(dim=1), vlabels)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / len(val_loader)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        metrics[epoch] = get_metrics(pred_vlabels, true_vlabels)
        metrics[epoch]["train_loss"] = avg_loss
        metrics[epoch]["val_loss"] = avg_vloss
        print(metrics[epoch])
        
        #wandb.log({"avg_eloss": avg_loss, "avg_vloss": avg_vloss})

        # Log the running loss averaged per batch
        # for both training and validation

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "cnn" / f"model_{emb_type}_{emb_dim}_{max_len}_{timestamp}_{epoch}")
            torch.save(model.state_dict(), model_path)

    all_metrics = []
    for k, v in list(metrics.items())[:-2]:
        out = [k]
        for metric in list(v.values())[:-2]:
            out.append(round(metric, 3)) if metric else out.append(None)
        all_metrics.append(out)

    print(f"RESULTS FOR TRAINING CNN WITH:\nemb type: {emb_type}\nemb dim: {emb_dim}\nmax length: {max_len}\npadding pos: {pad_pos}\nbatch size: {batch_size}\n\n\n")

    print(tabulate(all_metrics, headers=["Fold", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC", "train_loss", "val_loss"]))

    #wandb.finish()

if __name__ == "__main__":
    train(emb_type="glove", pad_pos="head", num_epochs=10, max_len=256, emb_dim=300, batch_size=1)
