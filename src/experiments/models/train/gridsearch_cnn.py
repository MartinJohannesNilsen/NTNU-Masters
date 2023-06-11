import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
import h5py
import click
from sklearn.metrics import confusion_matrix, roc_auc_score

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


class TextDatasetH5py(Dataset):
    def __init__(self, path):
        self.f = h5py.File(path, "r")

    def __len__(self):
        len(self.f["label"])
        
    def __getitem__(self, idx):
        embs = self.f["emb_tensor"][idx]
        label = self.f["label"][idx]

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


model_to_dim = {
    "glove_50": 50,
    "glove": 300,
    "fasttext": 300,
    "bert": 768
}


torch.manual_seed(0)
def grid_search(config):
    """
    Run a random search based on the parameters passed in the config dictionary
    The random search stores the best scoring trial for further testing on the test set

    Args:
        config (dict): Dictionary containing parameter space to search
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = TextClassifier(emb_dim=config['emb_dim'], dropout=config["dropout"]).to(device)

    emb_str = f"{config['emb_type']}_{model_to_dim[config['emb_type']]}" if config["emb_type"] != "glove_50" else config["emb_type"]

    base_path = Path(os.path.abspath(__file__)).parents[12] / "features" / "embeddings"
    print(f"base_path: {base_path}")
    train_path = base_path / f"train_sliced_stair_twitter_{emb_str}_{config['pad_pos']}_{config['max_len']}.h5"
    val_path = base_path / f"val_sliced_stair_twitter_{emb_str}_{config['pad_pos']}_{config['max_len']}.h5"

    print(f"base_path: {base_path}")
    print(f"train_path: {train_path}")
    print(f"val_path: {val_path}")
    
    train_set = TextDatasetH5py(train_path)
    val_set = TextDatasetH5py(val_path)

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
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            weighting = []
            for l in labels:
                if l == 0:
                    weighting.append(class_wts[0])
                else:
                    weighting.append(class_wts[1])

            inputs, labels, weighting = inputs.to(device), labels.to(float).to(device), torch.tensor(weighting).to(device)
            criterion = nn.BCELoss(weight=weighting)

            optimizer.zero_grad()

            outputs = model(inputs).to(float)
            loss = criterion(outputs.squeeze(dim=1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)

        # Validation step
        val_loss = 0.0
        pred_vlabels = []
        true_vlabels = []
        with torch.no_grad():
            for i, vdata in enumerate(val_loader, 0):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(float).to(device)

                voutputs = model(vinputs).to(float)

                vweighting = []
                for l in vlabels:
                    if l == 0:
                        vweighting.append(class_wts[0])
                    else:
                        vweighting.append(class_wts[1])
                vweighting = torch.tensor(vweighting).to(device)

                criterion = nn.BCELoss(weight=vweighting)
                vloss = criterion(voutputs.squeeze(dim=1), vlabels)

                voutputs = voutputs.cpu()
                vlabels = vlabels.cpu()
                [true_vlabels.append(vlabel) for vlabel in vlabels]
                [pred_vlabels.append(1) if pred > 0.5 else pred_vlabels.append(0) for pred in voutputs[0]]

                val_loss += vloss.item()

        avg_vloss = val_loss/len(val_loader)    

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        gs_path = f"CNN_gridsearch_sliced_stair_twitter_{config['emb_type']}_{config['emb_dim']}_{config['pad_pos']}_{config['max_len']}"
        os.makedirs(gs_path, exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), f"{gs_path}/checkpoint.pt")
        checkpoint = Checkpoint.from_directory(gs_path)

        metrics = get_metrics(pred_vlabels, true_vlabels)
        # Append some additional information to metrics dict for convenience
        metrics["epoch"] = epoch
        metrics["avg_train_loss"] = avg_loss
        metrics["loss"] = avg_vloss
        print(f"metrics:\n{metrics}")
        session.report(metrics, checkpoint=checkpoint)

    print("Finished Training")


def test_best_model(best_result):
    """
    Test model with best configs on test set

    Args:
        best_result (dict): Dictionary containing best hyperparams and args for model
    """
    config = best_result.config # Fetch best config from saved checkpoints
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model = TextClassifier(emb_dim=config['emb_dim'], dropout=config["dropout"]).to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    print(f"checkpoint_path: {checkpoint_path}")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    emb_str = f"{config['emb_type']}_{model_to_dim[config['emb_type']]}" if config["emb_type"] != "glove_50" else config["emb_type"]
    base_path = Path(os.path.abspath(__file__)).parents[4] / "features" / "embeddings"
    test_path = base_path / f"test_sliced_stair_twitter_{emb_str}_{config['pad_pos']}_{config['max_len']}.h5"

    test_set = TextDatasetH5py(test_path)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, pin_memory=True) 

    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_trained_model(inputs).squeeze()

            outputs = outputs.cpu()
            labels = labels.cpu()

            [true_labels.append(label) for label in labels]
            [pred_labels.append(1) if pred > 0.5 else pred_labels.append(0) for pred in outputs]

    metrics = get_metrics(pred_labels, true_labels)
    print(f"Best results with config:\n{config}")
    print(f"Got metrics: {metrics}")


@click.command()
@click.option("-e", "--emb_type", type=click.Choice(["fasttext", "glove", "bert", "glove_50"]) , help="Embedding type to be used for training")
@click.option("-l", "--max_len", type=click.INT, help="Max length of sentence to be allowed. Determines padding and truncation")
@click.option("-p", "--pad_pos", type=click.STRING, help="Position to place padding if necessary")
def main(emb_type: str, max_len: int, pad_pos: str, num_samples=10, max_num_epochs=10):
    """
    Run an entire random search for specified embedding type, max length and padding position.
    Results on test set are given for best model.

    Args:
        emb_type (str): Embedding type
        max_len (int): Max length
        pad_pos (str): Padding position
        num_samples (int, optional): Amount of trials for random search
        max_num_epochs (int, optional): Max number of epochs per trial
    """
    emb_dim = model_to_dim[emb_type]

    # Parameter space to search
    config = {
        "emb_dim": tune.choice([emb_dim]),
        "dropout": tune.choice([0.3, 0.4, 0.5, 0.6]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([64, 128, 256]),
        "max_len": max_len,
        "emb_type": emb_type,
        "pad_pos": pad_pos
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

    num_gpus = torch.cuda.device_count()
    print(f"num_gpus: {num_gpus}")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(grid_search),
            resources={"cpu": 1} # {"gpu": num_gpus} # This line is commented out for testing purposes. Use gpus for actual training
        ),
        tune_config=tune.TuneConfig(
            metric="f1_score",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=RunConfig(
            local_dir=str(Path(os.path.abspath(__file__)).parents[0] / "gs_results" / "cnn"),
            name=f"model_{emb_str}_{max_len}",
            progress_reporter=custom_reporter
        ),
        param_space=config
    )
    results = tuner.fit()
    
    # Best result is based on highest f2 score
    best_result = results.get_best_result("f2_score", "max")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result)

if __name__ == "__main__":
    main()