from torch.utils.data import DataLoader, Dataset
import h5py

class LSTMTextDatasetH5py(Dataset):
    def __init__(self, emb_path, max_len: int, pad_pos: str, liwc_path = None):
        self.emb_f = h5py.File(emb_path, "r")
        self.max_len = max_len
        self.pad_pos = pad_pos
        if liwc_path:
            self.liwc_f = h5py.File(liwc_path, "r")
            self.liwc_dim = self.liwc_f["emb_tensor"][0].shape[0]

    def __len__(self):
        return len(self.emb_f["label"])

    def __getitem__(self, i):
        embs = self.emb_f["emb_tensor"][i]
        label = self.emb_f["label"][i]
        length = self.emb_f["length"][i]

        start = 0
        end = start + (length - 1)
        if self.pad_pos == "head":
            start = -length
            end = -1
        elif self.pad_pos == "split":
            req_padding = self.max_len - length
            start = math.floor(req_padding/2)
            end = start + length - 1

        if liwc_path:
            liwc_scores = self.liwc_f["emb_tensor"][i]
            return embs, liwc_scores, label, start, end

        return embs, label, start, end
    
    def get_class_weights(self):
        labels = self.emb_f["label"]

        total_texts = labels.shape[0]
        num_shooter_texts = sum(labels)
        num_non_shooter_texts = total_texts - num_shooter_texts
        non_shooter_wt = total_texts / num_non_shooter_texts
        shooter_wt = total_texts / num_shooter_texts

        print(f"non_shooter: {non_shooter_wt}\nshooter: {shooter_wt}")

        return [non_shooter_wt, shooter_wt]


class CNNTextDatasetH5py(Dataset):
    def __init__(self, emb_path, liwc_path = None):
        self.emb_f = h5py.File(emb_path, "r")
        self.liwc_path = liwc_path
        if liwc_path:
            self.liwc_f = h5py.File(liwc_path, "r")
            self.liwc_dim = self.liwc_f["emb_tensor"][0].shape[0]

    def __len__(self):
        return 200 #len(self.emb_f["label"])
        
    def __getitem__(self, i):
        embs = self.emb_f["emb_tensor"][i]
        label = self.emb_f["label"][i]
        if self.liwc_path:
            liwc_scores = self.liwc_f["emb_tensor"][i]
            return embs, liwc_scores, label

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