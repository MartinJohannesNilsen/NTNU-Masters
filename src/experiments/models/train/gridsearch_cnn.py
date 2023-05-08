import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from pathlib import Path
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
import h5py
from experiments.utils.metrics import get_metrics
import click


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
        return len(self.f["label"])

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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #search_folder = create_search_run()

    model = TextClassifier(emb_dim=config['emb_dim'], dropout=config["dropout"]).to(device)

    base_path = Path(os.path.abspath(__file__)).parents[3] / "experiments" / "features" / "embeddings" / "new"
    train_path = base_path / f"train_sliced_stair_twitter_{config['emb_type']}_{config['emb_dim']}_{config['pad_pos']}_{config['max_len']}.h5"
    val_path = base_path / f"val_sliced_stair_twitter_{config['emb_type']}_{config['emb_dim']}_{config['pad_pos']}_{config['max_len']}.h5"

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

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], momentum=0.9)
    class_wts = train_set.get_class_weights()

    for epoch in range(10):  # loop over the dataset multiple times
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

            inputs, labels, weighting = inputs.to(device), labels.to(device), weighting.to(device)
            criterion = nn.BCELoss(weight=weighting)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            
        avg_loss = running_loss / len(train_loader)

        # Validation loss
        val_loss = 0.0
        pred_vlabels = []
        true_vlabels = []
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                vinputs, vlabels = data
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)

                voutputs = model(inputs)
                [true_vlabels.append(vlabel) for vlabel in vlabels]
                [pred_vlabels.append(1) if pred > 0.5 else pred_vlabels.append(0) for pred in voutputs[0]]

                vweighting = []
                for l in vlabels:
                    if l == 0:
                        vweighting.append(class_wts[0])
                    else:
                        vweighting.append(class_wts[1])

                criterion = nn.BCELoss(weight=vweighting)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

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
        metrics["epoch"] = epoch
        metrics["avg_train_loss"] = avg_loss
        metrics["loss"] = avg_vloss
        session.report(metrics, checkpoint=checkpoint)

    print("Finished Training")


def test_best_model(best_result):
    cfig = best_result.config
    best_trained_model = TextClassifier(cfig["emb_dim"], cfig["dropout"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    base_path = Path(os.path.abspath(__file__)).parents[3] / "experiments" / "features" / "embeddings" / "new"
    test_path = base_path / f"test_sliced_stair_twitter_{cfig['emb_type']}_{cfig['emb_dim']}_{cfig['pad_pos']}_{cfig['max_len']}.h5"

    test_set = TextDatasetH5py(test_path)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, pin_memory=True) 

    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            [true_labels.append(label) for label in labels]
            [pred_labels.append(1) if pred > 0.5 else pred_labels.append(0) for pred in outputs[0]]

    metrics = get_metrics(pred_labels, true_labels)
    print(f"Best results with config:\n{cfig}")
    print(f"Got metrics: {metrics}")



@click.command()
@click.option("-e", "--emb_type", type=click.Choice(["fasttext", "glove", "bert", "glove_50"]) , help="Embedding type to be used for training")
@click.option("-l", "--max_len", type=click.INT, help="Max length of sentence to be allowed. Determines padding and truncation")
@click.option("-p", "--pad_pos", type=click.STRING, help="Position to place padding if necessary")
def main(emb_type: str, max_len: int, pad_pos: str, num_samples=10, max_num_epochs=10):
    emb_dim = model_to_dim[emb_type]

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
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(grid_search),
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result)

if __name__ == "__main__":
    main()