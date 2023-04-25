import sys 
import os
from pathlib import Path
import torch
import sys
import csv
# sys.exit(1)
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
# src/experiments/utils
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"
import h5py
import numpy as np
from experiments.utils.metrics import get_metrics
from tabulate import tabulate

# Maxsize of csv field size
def _find_field_size_limit():
    max_int = sys.maxsize
    while True:
        # Decrease the value by factor 10 as long as the OverflowError occurs.
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int/10)


class TextDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.file = h5py.File(self.data_path, "r")
        self.data_len = self.file["idx"].shape[0]
        #print(f"Data length: {self.data_len}")


    def __len__(self):
        return self.data_len


    def __getitem__(self, idx):
        features = torch.from_numpy(self.file["emb_tensor"][idx])
        labels = self.file["label"][idx]

        return features, labels
    

    def get_class_weights(self):
        labels = self.file["label"]

        total_texts = labels.shape[0]
        num_shooter_texts = sum(labels)
        num_non_shooter_texts = total_texts - num_shooter_texts

        non_shooter_wt = total_texts / num_non_shooter_texts
        shooter_wt = total_texts / num_shooter_texts

        print(f"non_shooter: {non_shooter_wt}\nshooter: {shooter_wt}")

        return [non_shooter_wt, shooter_wt]





class LSTMTextClassifier(nn.Module):
    def __init__(self, batch_size: int = None, emb_dim: int = 300, sentence_len: int = 256, dropout: int = 0.5, hidden_size: int = 128, num_layers: int = 2):
        super(LSTMTextClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(sentence_len, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(2*hidden_size, 1)
        #self.dropout = nn.Dropout(p=dropout)
        
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

        # Init hidden layer and cell states for each forward pass

        hidden_states = torch.zeros(self.num_layers, x.size[0], self.hidden_size)
        cell_states = torch.zeros(self.num_layers, x.size[0], self.hidden_size)

        out, _ = self.lstm(x, (hidden_states, cell_states))
        out = self.fc(out[:, -1, :])

        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0) 

def train(embedding_type: str, pad_pos: str = "tail", num_epochs: int = 10, sentence_length: int = 256, embedding_dim: int = 300):
    # 222

    # Read data
    base_path = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings"

    emb_str = f"{embedding_type}" if embedding_dim == 300 else f"{embedding_type}_{embedding_dim}"
    sent_len_str = "" if sentence_length == 512 else f"_{sentence_length}"


    train_path = base_path / f"train_sliced_stair_twitter_{emb_str}_{pad_pos}{sent_len_str}.h5"
    val_path = base_path / f"test_sliced_stair_twitter_{emb_str}_{pad_pos}{sent_len_str}.h5"

    # Creating datasets for use with dataloaders
    train_set = TextDataset(train_path)
    val_set = TextDataset(val_path)

    # Load dataset
    train_loader = DataLoader(train_set, batch_size=222, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)

    # Create model
    model = LSTMTextClassifier(emb_dim=embedding_dim).to(device)

    # Create loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    class_wts = train_set.get_class_weights() # Make class wts proportional to proportion of class occurences

    # Run epoch of 
    def run_epoch():
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(train_loader):
            inputs, labels = data
            labels = labels.to(torch.float32)
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(f"Shape of input tensor: {inputs.shape}")
            optimizer.zero_grad()

            outputs = model(inputs)
            #print(f"out: {outputs}")

            weighting = []
            for l in labels:
                if l == 0:
                    weighting.append(class_wts[0])
                else:
                    weighting.append(class_wts[1])

            loss_fn = nn.BCELoss(weight=torch.tensor(weighting))

            loss = loss_fn(outputs.squeeze(), labels.to(torch.float32)) # Unsqueeze target tensor to allow for batching and same dims for out and target
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            
            # Update reported loss values every 50 steps
            if i % 50 == 49:
                last_loss = running_loss / 50
                running_loss = 0.
            
        return last_loss


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

    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = run_epoch()

        # We don't need gradients on to do reporting
        model.train(False)

        pred_vlabels = []
        true_vlabels = []

        running_vloss = 0.0
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)

            [true_vlabels.append(l) for l in vlabels]
            [pred_vlabels.append(1) if pred > 0.5 else pred_vlabels.append(0) for pred in voutputs[0]]
            
            weighting = []
            for vl in vlabels:
                if vl == 0:
                    weighting.append(class_wts[0])
                else:
                    weighting.append(class_wts[1])

            loss_fn = nn.BCELoss(weight=torch.tensor(weighting))
            vloss = loss_fn(voutputs, vlabels.to(torch.float32).unsqueeze(1))
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        metrics[epoch] = get_metrics(pred_vlabels, true_vlabels)
        matrics[epoch]["train_loss"] = avg_loss
        matrics[epoch]["val_loss"] = avg_vloss
        print(metrics[epoch])
        
        #wandb.log({"avg_eloss": avg_loss, "avg_vloss": avg_vloss})

        # Log the running loss averaged per batch
        # for both training and validation

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "cnn" / f"model_{embedding_type}_{embedding_dim}_{sentence_length}_{timestamp}_{epoch_number}")
            torch.save(model.state_dict(), model_path)

    all = []
    for k, v in metrics.items():
        out = [k]
        for metric in v.values():
            out.append(round(metric, 3)) if metric else out.append(None)
        all.append(out)

    print(f"RESULTS FOR TRAINING CNN WITH:\nemb type: {embedding_type}\nemb dim: {embedding_dim}\nsentence length: {sentence_length}\npadding pos: {pad_pos}\nbatch size: {222}\n\n\n")

    print(tabulate(all, headers=["Fold", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC", "train_loss", "val_loss"]))

    train_set.file.close()
    val_set.file.close()
    #wandb.finish()

if __name__ == "__main__":
    train(embedding_type="glove", pad_pos="head", num_epochs=10, sentence_length=256, embedding_dim=300)

