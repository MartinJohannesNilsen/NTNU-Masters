import sys 
import os
from pathlib import Path
import torch
import sys
import csv
import pandas as pd
# sys.exit(1)
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
# src/experiments/utils
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from datetime import datetime
import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"
import h5py
import numpy as np
from experiments.utils.metrics import get_metrics
from tabulate import tabulate
from experiments.utils.word_embeddings import get_emb_layer, get_glove_model, get_ft_model, get_id_from_tokens, _pad_and_get_orig_seq_len
from csv import QUOTE_NONE

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
    def __init__(self, df, emb_type, emb_dim):
        self.df = df
        #print(f"Data length: {self.data_len}")
        emb_model = None

        if emb_type == "glove":
            size = emb_dim == 50
            emb_model = get_glove_model(size)
        else:
            emb_model = get_ft_model()

        self.df["text"] = self.df["text"].map(lambda a: get_id_from_tokens(a, emb_model))


    def __len__(self):
        return len(self.df.index)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = row["text"]
        label = row["label"]

        return tokens, label, len(tokens)
    

    def get_class_weights(self):
        total_texts = self.__len__()
        num_shooter_texts, num_non_shooter_texts = self.df["label"].value_counts()

        print(f"Value counts:\n{self.df['label'].value_counts()}")

        non_shooter_wt = total_texts / num_non_shooter_texts
        shooter_wt = total_texts / num_shooter_texts

        print(f"non_shooter: {non_shooter_wt}\nshooter: {shooter_wt}")

        return [non_shooter_wt, shooter_wt]


class LSTMTextClassifier(nn.Module):
    def __init__(self, embs, vocab, batch_size: int = None, emb_dim: int = 300, sentence_len: int = 256, dropout: int = 0.5, hidden_size: int = 128, num_layers: int = 2):
        super(LSTMTextClassifier, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs).float())

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(sentence_len, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(2*hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout)
        
        self.sig = nn.Sigmoid() # Sigmoid to squeeze final vals between 0 and 1 to accomodate for binary class prob



    def forward(self, x, length):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Init hidden layer and cell states for each forward pass

        """ hidden_states = torch.zeros(self.num_layers, x.size[0], self.hidden_size)
        cell_states = torch.zeros(self.num_layers, x.size[0], self.hidden_size) """


        embs = self.embedding(x)

        #padded_embs = pad_sequence(embs, batch_first=True)

        packed_input = pack_padded_sequence(embs, length, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_input)

        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        out_forward = out[range(len(out)), length - 1, :self.hidden_size] # Forward dependencies
        out_backwards = out[:, 0, self.hidden_size:] # Backward dependencies

        out_reduced = torch.cat((out_forward, out_backwards), 1) # Concat for fc layer and final pred
        out_dropped = self.dropout(out_reduced) # Dropout layer

        logit = self.fc(out_dropped)
        logit = torch.squeeze(logit, 1)

        pred = self.sig(logit)

        return pred


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

def train(embedding_type: str, pad_pos: str = "tail", num_epochs: int = 10, sentence_length: int = 256, embedding_dim: int = 300):
    # 222

    # Read data
    base_path = Path(os.path.abspath(__file__)).parents[3] / "dataset_creation" / "data" / "train_test_preprocessed"

    sent_len_str = "" if sentence_length == 512 else f"_{sentence_length}"


    train_df = pd.read_csv(base_path / f"train_sliced_stair_twitter{sent_len_str}_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    test_df = pd.read_csv(base_path / f"test_sliced_stair_twitter{sent_len_str}_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

    # Creating datasets for use with dataloaders
    train_set = TextDataset(train_df, emb_type=embedding_type, emb_dim=embedding_dim)
    test_set = TextDataset(test_df, emb_type=embedding_type, emb_dim=embedding_dim)

    # Load dataset
    train_loader = DataLoader(train_set, batch_size=222, shuffle=False, pin_memory=True)
    val_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    vocab, embs = get_vocab_embs(embedding_dim, embedding_type)

    # Create model
    model = LSTMTextClassifier(embs=embs, vocab=vocab, emb_dim=embedding_dim).to(device)

    # Create loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    class_wts = train_set.get_class_weights() # Make class wts proportional to proportion of class occurences

    # Run epoch of 
    def run_epoch():
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(train_loader):
            inputs, labels, lengths = data
            labels = labels.to(torch.float32)
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
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
            vinputs, vlabels, vlengths = vdata
            voutputs = model(vinputs, vlengths)

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

    all_metrics = []
    for k, v in metrics.items():
        out = [k]
        for metric in v.values():
            out.append(round(metric, 3)) if metric else out.append(None)
        all.append(out)

    print(f"RESULTS FOR TRAINING CNN WITH:\nemb type: {embedding_type}\nemb dim: {embedding_dim}\nsentence length: {sentence_length}\npadding pos: {pad_pos}\nbatch size: {222}\n\n\n")

    print(tabulate(all_metrics, headers=["Fold", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC", "train_loss", "val_loss"]))

    train_set.file.close()
    val_set.file.close()
    #wandb.finish()


# Helper functions for feature based training
SUPPORTED_EMBEDDINGS = ["glove", "glove_50", "fasttext", "bert"]
def _embedding(path, emb_type = "glove", batch_size = None, cross_validation_splits: int = None):
    assert emb_type in SUPPORTED_EMBEDDINGS, "Embedding type not supported!"
    training(saved_model_dir=Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'svm' / 'embeddings' / emb_type / Path(path).stem, path=path, batch_size=batch_size, cross_validation_splits=cross_validation_splits)

SUPPORTED_LIWC_DICTS = ["2022", "2015", "2007", "2001"]
def _liwc(path, liwc_dict = "2022", batch_size = None, cross_validation_splits: int = None):
    assert liwc_dict in SUPPORTED_LIWC_DICTS, "LIWC dictionary version not supported!"
    training(saved_model_dir=Path(os.path.abspath(__file__)).parents[1] / 'saved_models' / 'svm' / 'liwc' / f'{liwc_dict}' / Path(path).stem, path=path, batch_size=batch_size, cross_validation_splits=cross_validation_splits)


# Training based on selected feature
def train_embeddings(path: str, emb:str = "glove", cross_val_splits = None):
    assert emb in SUPPORTED_EMBEDDINGS, "Embedding not supported!"
    if cross_val_splits:
        _embedding(emb_type=emb, path=path, cross_validation_splits=cross_val_splits)
    else:
        _embedding(emb_type=emb, path=path, batch_size=32)


def train_liwc(path: str, liwc_dict:str = "2022", cross_val_splits = None):
    assert liwc_dict in SUPPORTED_LIWC_DICTS, "Liwc dictionary not supported!"
    if cross_val_splits:
        _liwc(path=path, liwc_dict=liwc_dict, cross_validation_splits=cross_val_splits)
    else:
        _liwc(path=path, liwc_dict=liwc_dict)


""" click.option = partial(click.option, show_default=True)
@click.command()
@click.argument("path", nargs=1)
def main(path):

    # Check that path leads to file
    assert os.path.isfile(path), "No file found!"

    if "embeddings" in path:
        # Find emb_type
        emb_type = None
        if "glove_50" in path:
            emb_type = "glove_50"
        elif "glove" in path:
            emb_type = "glove"
        elif "fasttext" in path:
            emb_type = "fasttext"
        elif "bert" in path:
            emb_type = "bert"
        assert emb_type, "Incorrect format, could not find embedding!"
        train_embeddings(path, emb_type)
        
    elif "LIWC" in path:
        # Find liwc_dict
        liwc_dict = None
        if "2022" in path:
            liwc_dict = "2022"
        elif "2015" in path:
            liwc_dict = "2015"
        elif "2007" in path:
            liwc_dict = "2007"
        elif "2001" in path:
            liwc_dict = "2001"
        assert liwc_dict, "Incorrect format, could not find LIWC dict!"
        train_liwc(path, liwc_dict) """

if __name__ == "__main__":
    train(embedding_type="glove", pad_pos="head", num_epochs=10, sentence_length=256, embedding_dim=300)

