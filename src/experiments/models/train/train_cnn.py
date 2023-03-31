import sys 
import os
from pathlib import Path
import pandas as pd
import torch
from csv import QUOTE_NONE
import sys
import csv
# sys.exit(1)
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
# src/experiments/utils
from utils.word_embeddings import preprocess_text, get_glove_word_vectors
from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import wandb
import random
from typing import Type
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    def __init__(self, embeddings, labels, train=False):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        if len(self.labels) == len(self.embeddings):
            return len(self.labels)
        else:
            return -1

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1,1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(2664, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flat(x)
        x = self.fc3(x)
        x = self.sig(x)

        return x[0]


def train():

    # Read data
    base_path = Path(os.path.abspath(__file__)).parents[3] / "dataset_creation" / "data"
    datasets = {
        "school_shooters": base_path / "school_shooters.csv",
        "manifestos": base_path / "manifestos.csv",
        "stair_twitter_archive": base_path / "stair_twitter_archive.csv",
        "twitter": base_path / "twitter.csv",
        "stream_of_consciousness": base_path / "stream_of_consciousness.csv"
    }
    schoolshootersinfo_df = pd.read_csv(datasets["school_shooters"], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)
    stair_twitter_archive_df = pd.read_csv(datasets["stair_twitter_archive"], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)
    twitter_df = pd.read_csv(datasets["twitter"], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)
    stream_of_consciousness_df = pd.read_csv(datasets["stream_of_consciousness"], encoding="utf-8", delimiter="‎", engine="python", quoting=QUOTE_NONE)

    # Set label of school shooter or not
    # 0 = NOT school shooter
    # 1 = school shooter
    schoolshootersinfo_df["label"] = 1
    stair_twitter_archive_df["label"] = 1
    twitter_df["label"] = 0
    stream_of_consciousness_df["label"] = 0

    # Create shooter vs non-shooter dfs
    shooter_df = pd.concat([schoolshootersinfo_df[:100], stair_twitter_archive_df[:100]], ignore_index=True)
    non_shooter_df = pd.concat([twitter_df[:100], stream_of_consciousness_df[:100]], ignore_index=True)
    whole_corpus_df = pd.concat([shooter_df, non_shooter_df], ignore_index=True).sample(frac=1)

    # Preprocess text into tokens
    whole_corpus_df["text"] = whole_corpus_df["text"].map(lambda a: preprocess_text(a))

    # In some cases the text is entirely removed by preprocessing. 
    # Drop rows where this is the case
    whole_corpus_df = whole_corpus_df[whole_corpus_df["text"].map(len) > 0]

    sentence_lengths = [len(t) for t in whole_corpus_df["text"]]
    max_len = max(sentence_lengths)

    # Find max sentence length and send to embeddings method
    # This will return all sentences converted to a matrix of glove word embeddings
    # Each sentence is padded to the same length as max_len to facilitate use in the neural net
    whole_corpus_df["text"] = whole_corpus_df["text"].map(lambda a: get_glove_word_vectors(a, sentence_length=max_len))

    # Pickle
    # print(Path(os.path.abspath("")).parents[1] / "dataset_creation" / "data" / "all_data_glove_emb.pkl")
    # whole_corpus_df.to_pickle(Path(os.path.abspath("")).parents[1] / "dataset_creation" / "data" / "all_data_glove_emb.pkl")

    x_train, x_test, y_train, y_test = train_test_split(whole_corpus_df["text"], whole_corpus_df["label"], test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    # Creating datasets for use with dataloaders
    train_set = TextDataset(x_train.to_numpy(), y_train.to_numpy())
    val_set = TextDataset(x_val.to_numpy(), y_val.to_numpy())

    # Load dataset
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Create model
    model = TextClassifier().to(device)

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
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels.to(torch.float32))
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

    wandb.finish()

if __name__ == "__main__":
    _find_field_size_limit()
    train()
