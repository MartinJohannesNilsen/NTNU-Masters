import sys 
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
import torch
import csv
# sys.exit(1)
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
import click

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

    def __len__(self):
        return self.file["idx"].shape[0]

    def __getitem__(self, idx):
        features = torch.from_numpy(self.file["emb_tensor"][idx])
        label = self.file["label"][idx]

        return features, label
    
    def get_class_weights(self):
        labels = self.file["label"]
        total_texts = labels.shape[0]

        num_shooter_texts = sum(labels)
        num_non_shooter_texts = total_texts - num_shooter_texts
        non_shooter_wt = total_texts / num_non_shooter_texts
        shooter_wt = total_texts / num_shooter_texts

        print(f"non_shooter: {non_shooter_wt}\nshooter: {shooter_wt}")

        return [non_shooter_wt, shooter_wt]





class TextClassifier(nn.Module):
    def __init__(self, emb_dim: int = 300, filter_sizes = [3,4,5], num_filters = [100,100,100], dropout: int = 0.5):
        super(TextClassifier, self).__init__()

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(emb_type: str, pad_pos: str = "tail", num_epochs: int = 10, max_len: int = 256, stair_twitter: bool = True, batch_size: int = 64):

    # Read data
    stair_twitter_str = "sliced_stair_twitter" if stair_twitter else "no_stair_twitter"
    emb_dim = model_to_dim[emb_type]

    base_path = Path(os.path.abspath(__file__)).parents[3] / "experiments" / "features" / "embeddings" / "new"
    train_path = base_path / f"train_{stair_twitter_str}_{emb_type}_{emb_dim}_{pad_pos}_{max_len}.h5"
    val_path = base_path / f"val_{stair_twitter_str}_{emb_type}_{emb_dim}_{pad_pos}_{max_len}.h5"

    # Creating datasets for use with dataloaders
    train_set = TextDataset(train_path)
    val_set = TextDataset(val_path)

    # Load dataset
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, pin_memory=True)

    # Create model
    model = TextClassifier(emb_dim=emb_dim).to(device)

    # Create loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    class_wts = train_set.get_class_weights() # Make class wts proportional to proportion of class occurences

    # Run epoch of 
    def run_epoch():
        running_loss = 0.

        for _, data in enumerate(train_loader):
            inputs, labels = data
            labels = labels.to(torch.float32)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

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
                        
        return running_loss/len(train_loader)


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
        with torch.no_grad():
            for _, vdata in enumerate(val_loader):
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
                running_vloss += vloss.item()

        avg_vloss = running_vloss/len(val_loader)

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
            model_path = str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "cnn" / f"model_{emb_type}_{emb_dim}_{max_len}_{timestamp}_{epoch_number}")
            torch.save(model.state_dict(), model_path)

    all = []
    for k, v in metrics.items():
        out = [k]
        for metric in list(v.values())[:-2]:
            out.append(round(metric, 3)) if metric else out.append(None)
        all.append(out)

    print(f"RESULTS FOR TRAINING CNN WITH:\nemb type: {emb_type}\nemb dim: {emb_dim}\nsentence length: {max_len}\npadding pos: {pad_pos}\nbatch size: {222}\n\n\n")

    print(tabulate(all, headers=["Fold", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC", "train_loss", "val_loss"]))

    train_set.file.close()
    val_set.file.close()
    #wandb.finish()

@click.command()
@click.option("-e", "--emb", type=click.Choice(["fasttext", "glove", "bert"]) , help="Embedding type to be used for training")
@click.option("-d", "--dim", type=click.INT, help="Dimension of embeddings to be used")
@click.option("-l", "--length", type=click.INT, help="Max length of sentence to be allowed. Determines padding and truncation")
@click.option("-p", "--pad_pos", type=click.STRING, help="Position to place padding if necessary")
def main(emb, dim, length, pad_pos):
    train(emb_type=emb, pad_pos=pad_pos, num_epochs=10, max_len=length, emb_dim=dim)

if __name__ == "__main__":
    train(emb_type="bert", pad_pos="tail", num_epochs=10, max_len=256)
