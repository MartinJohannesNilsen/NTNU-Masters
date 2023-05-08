import sys 
import os
from pathlib import Path
import torch
import sys
import csv
import pandas as pd
import time
# sys.exit(1)
path1 = str(Path(os.path.abspath(__file__)).parents[3])
path2 = str(Path(os.path.abspath(__file__)).parents[2])
print(f"path1: {path1}")
print(f"path2: {path2}")
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


class LSTMTextClassifier(nn.Module):
    def __init__(self, embs, emb_dim: int = 300, dropout: int = 0.5, hidden_size: int = 128, num_layers: int = 2):
        super(LSTMTextClassifier, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs).float())
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

        embs = self.embedding(x)
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

def check_mem_usage():
    print("torch.cuda.memory_allocated: %fMB"%(torch.cuda.memory_allocated(0)/1024/1024))
    print("torch.cuda.memory_reserved: %fMB"%(torch.cuda.memory_reserved(0)/1024/1024))
    print("torch.cuda.max_memory_reserved: %fMB"%(torch.cuda.max_memory_reserved(0)/1024/1024))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0) 

def train(embedding_type: str, pad_pos: str = "tail", num_epochs: int = 10, sentence_length: int = 256, embedding_dim: int = 300, batch_size: int = 32):
    # 222

    # Read data
    base_path = None
    if embedding_type == "glove":
        base_path = Path(os.path.abspath(__file__)).parents[7] / "dataset_creation" / "data" / "train_test" / "new_preprocessed"
    elif embedding_type == "fasttext":
        base_path = Path(os.path.abspath(__file__)).parents[7] / "dataset_creation" / "data" / "train_test" / "new_preprocessed_nltk"

    print(f"Fetching data from {base_path}")

    train_df = pd.read_csv(base_path / f"train_sliced_stair_twitter_{sentence_length}_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    test_df = pd.read_csv(base_path / f"test_sliced_stair_twitter_{sentence_length}_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    val_df = pd.read_csv(base_path / f"val_sliced_stair_twitter_{sentence_length}_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

    print("Create vocab...")
    word_to_idx = create_vocab_w_idx(pd.concat([train_df, test_df, val_df], axis=0))
    vocab_len = len(word_to_idx)

    print("Convert words to ids and pad...")

    train_df["text"] = train_df["text"].map(lambda a: get_padded_ids(a, word_to_idx, pad_pos, sentence_length))
    val_df["text"] = val_df["text"].map(lambda a: get_padded_ids(a, word_to_idx, pad_pos, sentence_length))

    print("Create emb matrix")
    emb_mat = get_emb_matrix(embedding_dim, embedding_type, vocab_len, word_to_idx)

    print("Constructing model...")
    # Create model

    print("mem usage before constructing model")
    check_mem_usage()

    model = LSTMTextClassifier(embs=emb_mat, emb_dim=embedding_dim).to(device)

    print("mem usage after creating model")
    check_mem_usage()


    print("Garbage collect vocab dict and emb matrix")
    word_to_idx = None
    emb_mat = None

    print("Creating datasets...")

    # Creating datasets for use with dataloaders
    train_set = TextDataset(train_df)
    val_set = TextDataset(val_df)

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
            inputs, labels, lengths = data
            labels = labels.to(torch.float32).to(device)
            inputs = torch.from_numpy(np.array(inputs)).to(device)

            #print(f"Shape of input tensor: {inputs.shape}")
            optimizer.zero_grad()

            outputs = model(inputs, lengths)
            #print(f"out: {outputs}")

            weighting = []
            for l in labels:
                if l == 0:
                    weighting.append(class_wts[0])
                else:
                    weighting.append(class_wts[1])

            weighting = torch.tensor(weighting).to(device)
            loss_fn = nn.BCELoss(weight=weighting)
            loss = loss_fn(outputs.squeeze(dim=1), labels) # Unsqueeze target tensor to allow for batching and same dims for out and target
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        return running_loss / len(train_loader)


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
   
    metrics = {}

    print("Start training...")

    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = run_epoch()

        # We don't need gradients on to do reporting
        model.train(False)

        pred_vlabels = []
        true_vlabels = []

        running_vloss = 0.0

        print("validating")
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels, vlengths = vdata
                vinputs = torch.from_numpy(np.array(vinputs)).to(device)

                v_out = model(vinputs, vlengths)

                """ print(v_out)
                print(v_out[0]) """

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

        """ print(f"mem usage after epoch {epoch}")
        check_mem_usage() """
        
        #wandb.log({"avg_eloss": avg_loss, "avg_vloss": avg_vloss})

        # Log the running loss averaged per batch
        # for both training and validation

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "cnn" / f"model_{embedding_type}_{embedding_dim}_{sentence_length}_{timestamp}_{epoch_number}")
            torch.save(model.state_dict(), model_path)

    all_metrics = []
    for k, v in metrics.items():
        out = [k]
        for metric in list(v.values())[:-2]:
            out.append(round(metric, 3)) if metric else out.append(None)
        all_metrics.append(out)

    print(f"RESULTS FOR TRAINING CNN WITH:\nemb type: {embedding_type}\nemb dim: {embedding_dim}\nsentence length: {sentence_length}\npadding pos: {pad_pos}\nbatch size: {batch_size}\n\n\n")

    print(tabulate(all_metrics, headers=["Fold", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC", "train_loss", "val_loss"]))

    #wandb.finish()


if __name__ == "__main__":
    train(embedding_type="fasttext", pad_pos="tail", num_epochs=10, sentence_length=256, embedding_dim=300)
