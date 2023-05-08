import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import ray
from ray import tune
from ray.air import session, RunConfig
from ray.air.checkpoint import Checkpoint
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.experiment.trial import Trial
from pathlib import Path
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, roc_auc_score
import click
from csv import QUOTE_NONE
import nltk
from nltk.tokenize import RegexpTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#word_emb_path = str(Path(os.path.abspath(__file__)).parents[3] / "experiments" / "utils") 
p1 = str(Path(os.path.abspath(__file__)).parents[3])
p2 = str(Path(os.path.abspath(__file__)).parents[4])
sys.path.append(p2)
print(f"path2: {p1}")
sys.path.append(p1)

cache_dir = Path(os.path.abspath(__file__)).parents[4] / "resources" / ".vector_cache"
# Hyperparam tuning made from guide by https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html#tune-pytorch-cifar-ref

# UTILS NEEDED HERE. IMPORTS TOO BIG WITH RAY TUNE IF NOT

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
        "roc_auc": roc_auc
    }

    return metrics

def split_safe(text):
    return str(text).split(" ")

def create_vocab_w_idx(df: pd.DataFrame, is_preprocessed: bool = True, tk_type: str = "nltk"):
    """
    Input:
        df: Pandas dataframe containing a 'text' column

    Output:
        dict: A dictionary containing all unique words in vocab and their counts
    """
    if not is_preprocessed:
        if tk_type == "nltk":
            df["text"] = df["text"].map(lambda a: _tokenize_with_preprocessing(a, is_ft=True))
        else:
            df["text"] = df["text"].map(lambda a: _tokenize_with_preprocessing(a, is_ft=False))

    counts = {}
    for row in df["text"].map(lambda a: split_safe(a)):
        for word in row:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1

    # Drop underrepresented words and create vocab
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for key in counts.keys():
        if counts[key] >= 3:
            vocab[key] = len(vocab)

    return vocab


def get_emb_matrix(emb_dim, emb_type, vocab_len, word_to_index):

    path = None
    if emb_type == "glove" or "glove_50":
        dim_name = "6B.50d" if emb_dim == 50 else "840B.300d"
        path = cache_dir / f"glove.{dim_name}.txt"
    else:
        path = cache_dir / "wiki.en.vec"

    #embed_mat = np.random.rand(vocab_len + 1, emb_dim)
    embed_mat = np.zeros([vocab_len + 1, emb_dim])

    with open(path, "r+", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            elem = line.replace("\n", "").strip().split(" ")
            word = elem[0]

            if word not in word_to_index:
                continue

            word_idx = word_to_index[word]
            if word_idx <= vocab_len:
                embed_mat[word_idx] = np.asarray(elem[1:], dtype='float32')

    return embed_mat


def _tokenize_with_preprocessing(text: str, remove_url: bool = True, is_ft: bool = False):
    """For the embedders needing tokenized input. The method perform certain steps of text cleaning:
        - Stopword removal
        - Url replacement (token or removal)
        - Username removal
        - Hashtag removal
        - Character normalization

    Args:
        text (str): Input string.
        remove_url (bool, optional): Removes url completely. Defaults to True.

    Returns:
        List[str]: List of tokens
    """
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    tokenizer = None if is_ft else RegexpTokenizer("[\w']+")

    words = nltk.workd_tokenize(text) if is_ft else tokenizer.tokenize(text)

    url_replacement = "" if remove_url else "URLHYPERLINK"

    cleaned_words = []
    for word in words:
        word = word.lower()
        if word not in stopwords.words("english"):
            word = re.sub(r'http+|www+', url_replacement, word) # Replace urls with chosen string or remove completely
            word = re.sub(r'@[^ ]+', '', word) # Remove usernames in the context of Twitter posts
            word = re.sub(r'#', '', word) # Remove hashtags and keep words
            #word = re.sub(r'[^a-zA-Z0-9\s]', '', word) # ADDED AFTER WE MADE EMBS!!!!! Remove more special chars
            word = re.sub(r'([A-Za-z])\1{2,}', r'\1', word) # Character normalization, prevent words with letters repeated more than twice

            if word != "":
                cleaned_words.append(word)

    return cleaned_words


def get_id_from_tokens(text, word_to_idx):

    ids = []
    for token in split_safe(text):
        if token in word_to_idx:
            ids.append(word_to_idx[token])
        else:
            ids.append(1)

    return ids


def pad_ids(ids, pad_pos, max_len):
    length = len(ids)
    if length < max_len:
        req_padding = max_len - length
        pad = [0 for _ in range(req_padding)]

        if pad_pos == "head":
            pad += ids
            ids = pad

        elif pad_pos == "tail":
            ids += pad

        else:
            split_i = floor(req_padding/2)
            front_pad = [0 for _ in range(split_i)]
            end_pad = [0 for _ in range(req_padding - split_i)]
            front_pad += ids
            front_pad += end_pad
            ids = front_pad
            
    elif length > max_len:
        ids = ids[:max_len]
        length = max_len

    return np.array(ids), length


def get_padded_ids(text, word_to_idx, pad_pos, max_len):
    ids = get_id_from_tokens(text, word_to_idx)

    padded_ids, length = pad_ids(ids, pad_pos, max_len)

    return [padded_ids, length]


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

        print(f"shape out: {out.shape}")
        print(f"shape out_fw: {out_forward.shape}")
        print(f"shape out_bw: {out_backwards.shape}")

        out_reduced = torch.cat((out_forward, out_backwards), 1) # Concat for fc layer and final pred
        out_dropped = self.dropout(out_reduced) # Dropout layer
        out = self.fc(out_dropped)
        out = self.sig(out)

        return out


class TextDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return 50
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


model_to_dim = {
    "glove_50": 50,
    "glove": 300,
    "fasttext": 300,
    "bert": 768
}


torch.manual_seed(0)
def grid_search(config):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    emb_str = f"{config['emb_type']}_{model_to_dim[config['emb_type']]}" if config["emb_type"] != "glove_50" else config["emb_type"]
    base_path = None
    if config["emb_type"] != "fasttext":
        base_path = Path(os.path.abspath(__file__)).parents[7] / "dataset_creation" / "data" / "train_test" / "new_preprocessed"
    else:
        base_path = Path(os.path.abspath(__file__)).parents[7] / "dataset_creation" / "data" / "train_test" / "new_preprocessed_nltk"

    train_df = pd.read_csv(base_path / f"train_sliced_stair_twitter_{config['max_len']}_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    test_df = pd.read_csv(base_path / f"test_sliced_stair_twitter_{config['max_len']}_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    val_df = pd.read_csv(base_path / f"val_sliced_stair_twitter_{config['max_len']}_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

    word_to_idx = create_vocab_w_idx(pd.concat([train_df, test_df, val_df], axis=0))
    vocab_len = len(word_to_idx)
    train_df["text"] = train_df["text"].map(lambda a: get_padded_ids(a, word_to_idx, "tail", config["max_len"]))
    val_df["text"] = val_df["text"].map(lambda a: get_padded_ids(a, word_to_idx, "tail", config["max_len"]))
    emb_mat = get_emb_matrix(config["emb_dim"], config["emb_type"], vocab_len, word_to_idx)
    
    model = LSTMTextClassifier(embs=emb_mat, emb_dim=config["emb_dim"], hidden_size=config['hidden_size'], dropout=config["dropout"], num_layers=config["num_layers"]).to(device)
    
    word_to_idx = None
    emb_mat = None

    # Creating datasets for use with dataloaders
    train_set = TextDataset(train_df)
    val_set = TextDataset(val_df)

    # Load dataset
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)

    # To restore a checkpoint, use `session.get_checkpoint()`.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    class_wts = train_set.get_class_weights()

    for epoch in range(10):  # loop over the dataset multiple times
        print(f"epoch: {epoch}")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels, lengths = data
            weighting = []
            for l in labels:
                if l == 0:
                    weighting.append(class_wts[0])
                else:
                    weighting.append(class_wts[1])

            inputs, labels, weighting = inputs.to(device), labels.to(torch.float32).to(device), torch.tensor(weighting).to(device)
            criterion = nn.BCELoss(weight=weighting)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
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
                vinputs, vlabels, vlengths = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(torch.float32).to(device)

                voutputs = model(vinputs, vlengths)
                [true_vlabels.append(vlabel) for vlabel in vlabels]
                [pred_vlabels.append(1) if pred > 0.5 else pred_vlabels.append(0) for pred in voutputs[0]]

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

        avg_vloss = val_loss/len(val_loader)    

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        gs_path = f"LSTM_glove_ft_gridsearch_sliced_stair_twitter_{config['emb_type']}_{config['emb_dim']}_{config['max_len']}"
        os.makedirs(gs_path, exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), f"{gs_path}/checkpoint.pt")
        checkpoint = Checkpoint.from_directory(gs_path)

        epoch_metrics = get_metrics(pred_vlabels, true_vlabels)
        epoch_metrics["epoch"] = epoch
        epoch_metrics["avg_train_loss"] = avg_loss
        epoch_metrics["loss"] = avg_vloss
        print(f"metrics:\n{epoch_metrics}")
        session.report(epoch_metrics, checkpoint=checkpoint)

    print("Finished Training")


def test_best_model(best_result):
    config = best_result.config
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    emb_str = f"{config['emb_type']}_{model_to_dim[config['emb_type']]}" if config["emb_type"] != "glove_50" else config["emb_type"]
    base_path = None
    if config["emb_type"] != "fasttext":
        base_path = Path(os.path.abspath(__file__)).parents[7] / "dataset_creation" / "data" / "train_test" / "new_preprocessed"
    else:
        base_path = Path(os.path.abspath(__file__)).parents[7] / "dataset_creation" / "data" / "train_test" / "new_preprocessed_nltk"

    train_df = pd.read_csv(base_path / f"train_sliced_stair_twitter_{config['max_len']}_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    test_df = pd.read_csv(base_path / f"test_sliced_stair_twitter_{config['max_len']}_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")
    val_df = pd.read_csv(base_path / f"val_sliced_stair_twitter_{config['max_len']}_preprocessed.csv", sep="‎", quoting=QUOTE_NONE, engine="python")

    word_to_idx = create_vocab_w_idx(pd.concat([train_df, test_df, val_df], axis=0))
    vocab_len = len(word_to_idx)
    test_df["text"] = test_df["text"].map(lambda a: get_padded_ids(a, word_to_idx, "tail", config["max_len"]))
    emb_mat = get_emb_matrix(config["emb_dim"], config["emb_type"], vocab_len, word_to_idx)
        
    word_to_idx = None
    emb_mat = None
    train_df, val_df = None, None

    # Creating datasets for use with dataloaders
    test_set = TextDataset(test_df)

    # Load dataset
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, pin_memory=True)
    best_trained_model = LSTMTextClassifier(embs=emb_mat, emb_dim=config["emb_dim"], hidden_size=config['hidden_size'], dropout=config["dropout"], num_layers=config["num_layers"]).to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    print(f"checkpoint_path: {checkpoint_path}")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, lengths = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_trained_model(inputs, lengths).squeeze()

            [true_labels.append(label) for label in labels]
            [pred_labels.append(1) if pred > 0.5 else pred_labels.append(0) for pred in outputs]

    test_metrics = get_metrics(pred_labels, true_labels)
    print(f"Best results with config:\n{config}")
    print(f"Got metrics: {test_metrics}")


@click.command()
@click.option("-e", "--emb_type", type=click.Choice(["fasttext", "glove", "bert", "glove_50"]) , help="Embedding type to be used for training")
@click.option("-l", "--max_len", type=click.INT, help="Max length of sentence to be allowed. Determines padding and truncation")
def main(emb_type: str, max_len: int, num_samples=15, max_num_epochs=10):
    emb_dim = model_to_dim[emb_type]

    config = {
        "emb_dim": tune.choice([emb_dim]),
        "dropout": tune.choice([0.3, 0.4, 0.5, 0.6]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([64, 128, 256]),
        "max_len": max_len,
        "emb_type": emb_type,
        "hidden_size": tune.choice([64, 128, 256]),
        "num_layers": tune.choice([1, 2, 3])
    }
    scheduler = ASHAScheduler( 
        max_t=max_num_epochs,
        grace_period=3,
        reduction_factor=2)
    
    emb_str = f"{emb_type}_{model_to_dim[emb_type]}" if emb_type != "glove_50" else emb_type

    # Customize reporting to avoid excessive output printing from ray reporter
    class TrialTerminationReporter(CLIReporter):
        def __init__(self):
            super(TrialTerminationReporter, self).__init__()
            self.num_terminated = 0

        def should_report(self, trials, done=False):
            """Reports only on trial termination events."""
            old_num_terminated = self.num_terminated
            self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
            return self.num_terminated > old_num_terminated

    custom_reporter = TrialTerminationReporter()
    custom_reporter.add_metric_column("epoch")
    custom_reporter.add_metric_column("f1_score")
    custom_reporter.add_metric_column("avg_train_loss")
    custom_reporter.add_metric_column("loss")

    local_dir = str(Path(os.path.abspath(__file__)).parents[0] / "gs_results" / "lstm")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(grid_search),
            resources={"cpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="f1_score",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=RunConfig(
            local_dir=local_dir,
            name=f"model_{emb_str}_{max_len}",
            progress_reporter=custom_reporter
        ),
        param_space=config
    )
    results = tuner.fit()

    """ dfs = {result.log_dir: result.metrics_dataframe for result in results}
    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.mean_accuracy.plot(ax=ax, legend=False) """
    
    best_result = results.get_best_result("f1_score", "max")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result)

if __name__ == "__main__":
    main()