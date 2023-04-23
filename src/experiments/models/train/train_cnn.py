import sys 
import os
from pathlib import Path
import torch
from csv import QUOTE_NONE
import sys
import csv
# sys.exit(1)
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

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        features = torch.from_numpy(self.file["emb_tensor"][idx])
        label = self.file["label"][idx]

        return features, label


class TextClassifier(nn.Module):
    def __init__(self, batch_size: int, emb_dim: int = 300, sentence_len: int = 256, filter_sizes = [3,4,5], num_filters = [100,100,100], dropout: int = 0.5):
        super(TextClassifier, self).__init__()
        """ self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1,1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(2664, 1)
        self.sig = nn.Sigmoid() """

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
        """ x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flat(x)
        print(f"Size after flatten: {x.size()}")
        x = self.fc3(x)
        x = self.sig(x) """

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
        
        #print(f"Size after squeeze and flatten: {x_fc.size()}")

        # Compute logits. Output shape: (b, n_classes)
        out = self.fc(self.dropout(x_fc))

        #print(f"output after dropout={0.5} and fc layer: {out}")

        # Squeeze between class 0 and 1 (non-shooter and shooter)
        out = self.sig(out)

        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

def train():

    # Read data
    base_path = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings"

    train_path = base_path / "test_sliced_stair_twitter_glove_50_tail_256.h5"
    val_path = base_path / "hold_out_test_sliced_stair_twitter_glove_50_tail_256.h5"


    # Creating datasets for use with dataloaders
    train_set = TextDataset(train_path)
    val_set = TextDataset(val_path)

    # Load dataset
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)

    # Create model
    model = TextClassifier(batch_size=1, emb_dim=50).to(device)

    # Create loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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
            loss = loss_fn(outputs, labels.to(torch.float32).unsqueeze(1)) # Unsqueeze target tensor to allow for batching and same dims for out and target
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
    EPOCHS = 10
    best_vloss = 1_000_000.

    
    wandb.init(
        # set the wandb project where this run will be logged
        project="cnn-glove-features-predict-shooters",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "school-shooters-vs-non-school-shooters",
        "epochs": 10,
        }
    )
   

    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = run_epoch()

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels.to(torch.float32))
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')
        
        wandb.log({"avg_eloss": avg_loss, "avg_vloss": avg_vloss})

        # Log the running loss averaged per batch
        # for both training and validation

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = str(Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "cnn" / "glove_encodings" / f"model_{timestamp}_{epoch_number}")
            torch.save(model.state_dict(), model_path)

    train_set.file.close()
    val_set.file.close()
    wandb.finish()

if __name__ == "__main__":
    _find_field_size_limit()
    train()

