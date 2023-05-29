import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from pathlib import Path
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import h5py
from sklearn.metrics import confusion_matrix, roc_auc_score
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
from utils.extract_best_model_and_stats import get_best_scoring_config


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

    def forward(self, embs):
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
        x_reshaped = embs.permute(0, 2, 1)

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
    def __init__(self, emb_path):
        self.emb_f = h5py.File(emb_path, "r")
        print(f"emb length: {self.emb_f['label'].shape}")

    def __len__(self):
        return len(self.emb_f["label"])
        
    def __getitem__(self, i):
        embs = self.emb_f["emb_tensor"][i]
        label = self.emb_f["label"][i]

        return embs, label
    
    def get_class_weights(self):
        labels = self.emb_f["label"]

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
def train(emb_dim: int, dropout: float, emb_type: str, pad_pos: str, max_len: int, batch_size: int, lr: float):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #search_folder = create_search_run()

    emb_str = f"{emb_type}_{model_to_dim[emb_type]}" if emb_type != "glove_50" else emb_type

    base_path_embs = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings" / "new"
    train_path_embs = base_path_embs / f"train_sliced_stair_twitter_{emb_str}_{pad_pos}_{max_len}.h5"
    val_path_embs = base_path_embs / f"val_sliced_stair_twitter_{emb_str}_{pad_pos}_{max_len}.h5"
    
    train_set = TextDatasetH5py(train_path_embs)
    val_set = TextDatasetH5py(val_path_embs)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)

    model = TextClassifier(emb_dim=emb_dim, dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    class_wts = train_set.get_class_weights()

    for epoch in range(9):  # loop over the dataset multiple times
        print(f"epoch: {epoch}")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            emb_inputs, labels = data

            weighting = []
            for l in labels:
                if l == 0:
                    weighting.append(class_wts[0])
                else:
                    weighting.append(class_wts[1])

            emb_inputs, labels, weighting = emb_inputs.to(device), labels.to(float).to(device), torch.tensor(weighting).to(device)
            criterion = nn.BCELoss(weight=weighting)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(emb_inputs).to(float)
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
                vemb_inputs, vlabels = vdata
                vemb_inputs, vlabels = vemb_inputs.to(device), vlabels.to(float).to(device)

                voutputs = model(vemb_inputs).to(float)

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

        metrics = get_metrics(pred_vlabels, true_vlabels)
        metrics["epoch"] = epoch
        metrics["avg_train_loss"] = avg_loss
        metrics["loss"] = avg_vloss
        print(f"metrics:\n{metrics}")

    print("Finished Training")
    return model


def test_model(model, config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    emb_str = f"{config['emb_type']}_{model_to_dim[config['emb_type']]}" if config['emb_type'] != "glove_50" else config['emb_type']
    base_path_embs = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings"
    test_path_embs = base_path_embs / f"shooter_hold_out_{emb_str}_{config['pad_pos']}_{config['max_len']}.h5"

    test_set = TextDatasetH5py(test_path_embs)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, pin_memory=True) 

    true_labels = []
    pred_labels = []
    pred_values = []
    with torch.no_grad():
        for data in test_loader:
            emb_inputs, labels = data
            emb_inputs, labels = emb_inputs.to(device), labels.to(device)
            outputs = model(emb_inputs).to(float)

            outputs = outputs.cpu()
            labels = labels.cpu()

            [true_labels.append(label.item()) for label in labels]
            [pred_labels.append(1) if pred > 0.5 else pred_labels.append(0) for pred in outputs]
            [pred_values.append(output.item()) for output in outputs]

    metrics = get_metrics(pred_labels, true_labels)
    print(f"Best results with config:\n{config}")
    print(f"Got metrics: {metrics}")

    return pred_values, pred_labels, true_labels
 
if __name__ == "__main__":
    best_emb_type, best_config, best_score = get_best_scoring_config("cnn", "f2_score")
    
    print(f"Best emb type: {best_emb_type}")
    print(f"Best config:\n{best_config}")
    print(f"Best score:\n{best_score}")

    best_model = train(
        emb_dim=best_config["emb_dim"],
        dropout=best_config["dropout"],
        emb_type=best_config["emb_type"],
        pad_pos=best_config["pad_pos"],
        max_len=best_config["max_len"],
        batch_size=best_config["batch_size"],
        lr=best_config["lr"]
    )

    pred_values, pred_labels, true_labels = test_model(best_model, best_config)

    cnn_out_path = Path(os.path.abspath(__file__)).parents[2] / "pred_values" / "shooter_hold_out" / "cnn_preds.csv"
    cnn_df = pd.DataFrame({"pred_val": pred_values, "pred_label": pred_labels, "label": true_labels})
    cnn_df.to_csv(cnn_out_path, index=False)