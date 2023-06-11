import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import os
from torch.utils.data import DataLoader
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
from utils.extract_best_model_and_stats import get_best_scoring_config
from utils.metrics import get_metrics
from utils.cnn_model_structure import CNNTextClassifier
from utils.dataset_structures import CNNTextDatasetH5py
import pandas as pd
from pathlib import Path

# Hyperparam tuning made from guide by https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html#tune-pytorch-cifar-ref

model_to_dim = {
    "glove_50": 50,
    "glove": 300,
    "fasttext": 300,
    "bert": 768
}


torch.manual_seed(0)
def train(emb_dim: int, dropout: float, emb_type: str, pad_pos: str, max_len: int, batch_size: int, lr: float):
    """
    Complete training loop for CNN model taking embedding vectors and liwc vectors as inputs for each forward pass

    Args:
        emb_dim (int): Dimension of the embedding vector to be passed as input
        dropout (float): Amount of dropout to be applied
        emb_type (str): Type of embedding vector that is used. Can be glove, glove_50, fasttext or bert
        pad_pos (str): Position of padding applied to the embedding vector. Can be head, split or tail
        max_len (int): Max sequence length of an embedding vector
        batch_size (int): Batch size used in training
        lr (float): Learning rate

    Returns:
        Model (torch.nn.Module): Trained model combining embedding and liwc features
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    emb_str = f"{emb_type}_{model_to_dim[emb_type]}" if emb_type != "glove_50" else emb_type

    base_path_embs = Path(os.path.abspath(__file__)).parents[4] / "features" / "embeddings"
    train_path_embs = base_path_embs / f"train_sliced_stair_twitter_{emb_str}_{pad_pos}_{max_len}.h5"
    val_path_embs = base_path_embs / f"val_sliced_stair_twitter_{emb_str}_{pad_pos}_{max_len}.h5"

    base_path_liwc = Path(os.path.abspath(__file__)).parents[4] / "features" / "liwc" / "preprocessed" / "splits" / "h5" / "2022"
    train_path_liwc = base_path_liwc / f"train_sliced_stair_twitter_{max_len}_preprocessed.h5"
    val_path_liwc = base_path_liwc / f"val_sliced_stair_twitter_{max_len}_preprocessed.h5"

    print(f"base_path: {base_path_liwc}")
    print(f"train_path: {train_path_liwc}")
    print(f"val_path: {val_path_liwc}")
    
    # Create torch datasets. 
    # Datasets yield embedding vector, liwc vector and label for each call to next
    train_set = CNNTextDatasetH5py(train_path_embs, train_path_liwc)
    val_set = CNNTextDatasetH5py(val_path_embs, val_path_liwc)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)

    model = CNNTextClassifier(emb_dim=emb_dim, dropout=dropout, liwc_size=train_set.liwc_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    class_wts = train_set.get_class_weights() # Get weights for classes to compensate for minority class

    for epoch in range(10):
        print(f"epoch: {epoch}")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            emb_inputs, liwc_inputs, labels = data

            # BCELoss takes weights for each input, so we need to determine weight for each input given
            weighting = []
            for l in labels:
                if l == 0:
                    weighting.append(class_wts[0])
                else:
                    weighting.append(class_wts[1])

            emb_inputs, liwc_inputs, labels, weighting = emb_inputs.to(device), liwc_inputs.to(device), labels.to(float).to(device), torch.tensor(weighting).to(device)
            criterion = nn.BCELoss(weight=weighting)

            optimizer.zero_grad()
            outputs = model(emb_inputs, liwc_inputs).to(float)
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
                vemb_inputs, vliwc_inputs, vlabels = vdata
                vemb_inputs, vliwc_inputs, vlabels = vemb_inputs.to(device), vliwc_inputs.to(device), vlabels.to(float).to(device)

                voutputs = model(vemb_inputs, vliwc_inputs).to(float)

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
    """
    Test the best available model

    Args:
        model (torch.nn.Module): Best performing model
        config (dict): Dictionary containing configurations for the best performing model

    Returns:
        pred_values (List(float)): List of predicted values
        pred_labels (List(int)): List of predicted labels. These labels are calculated on the basis of pred_value > 0.5
        true_labels (List(int)): Actual labels
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    emb_str = f"{config['emb_type']}_{model_to_dim[config['emb_type']]}" if config['emb_type'] != "glove_50" else config['emb_type']
    base_path_embs = Path(os.path.abspath(__file__)).parents[2] / "features" / "embeddings"
    test_path_embs = base_path_embs / f"shooter_hold_out_{emb_str}_{config['pad_pos']}_{config['max_len']}.h5"

    base_path_liwc = Path(os.path.abspath(__file__)).parents[2] / "features" / "liwc" / "shooter_hold_out" / "h5" / "2022"
    test_path_liwc = base_path_liwc / f"shooter_hold_out_{config['max_len']}.h5"

    test_set = CNNTextDatasetH5py(test_path_embs, test_path_liwc)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, pin_memory=True) 

    true_labels = []
    pred_labels = []
    pred_values = []
    with torch.no_grad():
        for data in test_loader:
            emb_inputs, liwc_inputs, labels = data
            emb_inputs, liwc_inputs, labels = emb_inputs.to(device), liwc_inputs.to(device), labels.to(device)
            outputs = model(emb_inputs, liwc_inputs).to(float)

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
    best_emb_type, best_config, best_score = get_best_scoring_config("cnn_liwc", "f2_score")

    print(f"Best emb type: {best_emb_type}")
    print(f"Best config:\n{best_config}")
    print(f"Best score:\n{best_score}")

    # Train a model with the found best scoring configs
    best_model = train(
        emb_dim=best_config["emb_dim"],
        dropout=best_config["dropout"],
        emb_type=best_config["emb_type"],
        pad_pos=best_config["pad_pos"],
        max_len=best_config["max_len"],
        batch_size=best_config["batch_size"],
        lr=best_config["lr"]
    )

    # Test the best performing model on the test set
    pred_values, pred_labels, true_labels = test_model(best_model, best_config)

    # Store results from test run to compare with other architectures
    cnn_out_path = Path(os.path.abspath(__file__)).parents[2] / "pred_values" / "shooter_hold_out" / "cnn_w_liwc_preds.csv"
    cnn_df = pd.DataFrame({"pred_val": pred_values, "pred_label": pred_labels, "label": true_labels})
    cnn_df.to_csv(cnn_out_path, index=False)
    torch.save(best_model.state_dict(), Path(os.path.abspath(__file__)).parents[1] / "saved_models" / "best_cnn_emb_liwc.pt")
