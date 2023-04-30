import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import h5py as h5py
import sys
import os
from tabulate import tabulate
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
from utils.word_emb_utils import pad_input
import pandas as pd
from csv import QUOTE_NONE
from datetime import datetime
import numpy as np
from utils.metrics import get_metrics

# MODEL ARCHITECTURES

class LSTMTextClassifier(nn.Module):
    def __init__(self, emb_wts = None, emb_dim: int = 300, dropout: int = 0.5, hidden_size: int = 128, num_layers: int = 2):
        super(LSTMTextClassifier, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(emb_wts).float()) if emb_wts else None
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

        embs = pad_input(self.embedding(x), ) if self.embedding else x
        seq_len = length-1 if self.embedding else [(s-1) for s in length]

        packed_input = pack_padded_sequence(embs, length, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_input)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        out_forward = out[range(len(out)), seq_len, :self.hidden_size] # Get output of last LSTM node
        out_backwards = out[:, 0, self.hidden_size:] # Output of first node thingy thangy

        out_reduced = torch.cat((out_forward, out_backwards), 1) # Concat for fc layer and final pred
        out_dropped = self.dropout(out_reduced) # Dropout layer
        out = self.fc(out_dropped)
        out = self.sig(out)

        return out


# DATASETS
class TextDatasetH5py(Dataset):
    def __init__(self, path):
        self.f = h5py.File(path, "r")

    def __len__(self):
        return len(self.f["label"])

    def __getitem__(self, idx):
        embs = self.f["emb_tensor"][idx]
        label = self.f["label"][idx]
        length = self.f["length"][idx]

        return embs, label, length
    
    def get_class_weights(self):
        labels = self.f["label"]

        total_texts = labels.shape[0]
        num_shooter_texts = sum(labels)
        num_non_shooter_texts = total_texts - num_shooter_texts
        non_shooter_wt = total_texts / num_non_shooter_texts
        shooter_wt = total_texts / num_shooter_texts

        print(f"non_shooter: {non_shooter_wt}\nshooter: {shooter_wt}")

        return [non_shooter_wt, shooter_wt]
    

class TextDatasetDf(Dataset):
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


base_paths = {
    "pd": Path(os.path.abspath(__file__)).parents[3] / "dataset_creation" / "data" / "train_test" / "new_preprocessed",
    "h5": Path(os.path.abspath(__file__)).parents[3] / "experiments" / "features" / "embeddings" / "new"
}


model_to_dim = {
    "glove_50": 50,
    "glove": 300,
    "fasttext": 300,
    "bert": 768
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_with_premade_embs(emb_type: str, max_len: int = 256, num_epochs: int = 10, batch_size: int = 32, stair_twitter: bool = False, pad_pos: str = "tail"):
    # Read data

    stair_twitter_str = "sliced_stair_twitter" if stair_twitter else "no_stair_twitter"
    emb_dim = model_to_dim[emb_type]

    base_path = Path(os.path.abspath(__file__)).parents[3] / "experiments" / "features" / "embeddings" / "new"
    train_path = base_path / f"train_{stair_twitter_str}_{emb_type}_{emb_dim}_{pad_pos}_{max_len}.h5"
    val_path = base_path / f"val_{stair_twitter_str}_{emb_type}_{emb_dim}_{pad_pos}_{max_len}.h5"

    print("Creating datasets...")

    # Creating datasets for use with dataloaders
    train_set = TextDatasetH5py(train_path)
    val_set = TextDatasetH5py(val_path)

    print("Constructing dataloaders...")

    # Load dataset
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)

    # Create model
    model = LSTMTextClassifier(emb_dim=emb_dim, dropout=0.5, hidden_size=128, num_layers=2)

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
            lengths = torch.from_numpy(lengths).to(device)

            print(f"Shape of input tensor: {inputs.shape}")
            optimizer.zero_grad()
            outputs = model(inputs, lengths)

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
            
        return last_loss/len(train_loader)
  
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = 1_000_000.
    for epoch in range(num_epochs):
        print(f'EPOCH {epoch}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = run_epoch()
        model.train(False)

        print("validating...")

        pred_vlabels = []
        true_vlabels = []
        running_vloss = 0.0
        with torch.no_grad():
            for vdata in val_loader:
                vinputs, vlabels, vlengths = vdata
                vinputs = torch.from_numpy(np.array(vinputs)).to(device)

                [true_vlabels.append(vlabel) for vlabel in vlabels]
                [pred_vlabels.append(1) if pred > 0.5 else pred_vlabels.append(0) for pred in v_out[0]]

                vlabels = vlabels.to(torch.float32).to(device)
                v_out = model(vinputs, vlengths)
                
                weighting = [class_wts[l] for l in vlabels]
                weighting = torch.tensor(weighting).to(device)
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

        # Log the running loss averaged per batch for both training and validation

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

    print(f"RESULTS FOR TRAINING LSTM WITH:\nemb type: {emb_type}\nemb dim: {emb_dim}\nsentence length: {max_len}\npadding pos: {pad_pos}\nbatch size: {batch_size}\n\n\n")
    print(tabulate(all_metrics, headers=["Fold", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC", "train_loss", "val_loss"]))






def main(emb_type: str, max_len: int, num_epochs: int, batch_size: int, stair_twitter: bool, pad_pos: str, storage_type: str = "h5"):
    if storage_type == "h5":
        train_with_premade_embs(emb_type=emb_type, max_len=max_len, num_epochs=num_epochs, batch_size=batch_size, stair_twitter=stair_twitter, pad_pos=pad_pos)

if __name__ == "__main__":
    main()