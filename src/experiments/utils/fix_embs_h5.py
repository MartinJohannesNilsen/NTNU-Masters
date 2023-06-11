import h5py
from pathlib import Path
import os
import sys
sys.path.append(Path(os.path.abspath(__file__)).parents[0])
from create_and_store_embs import read_h5
import numpy as np

def first_time_setup_dataset(store, data):
        """
        First time dataset is accessed, it has to be created. Quick setup with dim0 = None to allow for resizing later
        """

        store.create_dataset("idx", compression="gzip", data=data["idx"], chunks=True, maxshape=(None, ))

        emb_data_shape = (None, data["emb_tensor"][0].shape[0], data["emb_tensor"][0].shape[1])

        print(f"emb_data_shape: {emb_data_shape}")
        print(f"len of first data sent to emb_data_column: {len(data['emb_tensor'])}")

        store.create_dataset("emb_tensor", compression="gzip", data=data["emb_tensor"], chunks=True, maxshape=emb_data_shape, dtype=np.float32)
        store.create_dataset("name", compression="gzip", data=data["name"], chunks=True, maxshape=(None, ))
        store.create_dataset("label", compression="gzip", data=data["label"], chunks=True, maxshape=(None, ))
        store.create_dataset("length", compression="gzip", data=data["length"], chunks=True, maxshape=(None, ))


def resize_and_append_datasets(store, data):
        """
        rows: Data to be stored
        chunk_size: Increment to increase size of dataset with
        Resize dim0 of dataset to fit new data
        """

        cols = ["idx", "emb_tensor", "name", "label", "length"]

        for colname in cols:
            store[colname].resize(store[colname].shape[0] + len(data[colname]), axis=0)
            store[colname][-len(data[colname]):] = data[colname]


def get_rows(store, start, step_size):
    idx_arr, embs_arr, name_arr, label_arr, length_arr = [], [], [], [], []
    for i in range(step_size):
        idx_arr.append(store["idx"][start+i])
        embs_arr.append(store["emb_tensor"][start+i])
        name_arr.append(store["name"][start+i])
        label_arr.append(store["label"][start+i])
        length_arr.append(store["length"][start+i])

    idx_arr = np.array(idx_arr)
    embs_arr = embs_arr = np.array(embs_arr)
    name_arr = np.array(name_arr, dtype=h5py.special_dtype(vlen=str))
    label_arr = np.array(label_arr, dtype=int)
    length_arr = np.array(length_arr, dtype=int)

    data = {
        "idx": idx_arr,
        "emb_tensor": embs_arr,
        "name": name_arr,
        "label": label_arr,
        "length": length_arr
    }

    return data
    
        

def fix_embeddings(old_fpath: str, idx_to_be_removed, step_size: int = 200):
    """
    Function to create and store embeddings to file with given step size. Helps alleviate memory constraints with large embeddings sizes

    df: Dataframe containing text from shooters
    fpath: path to file, file format should be h5 (hdf5)
    step_size: Amount of rows to be processed at once
    emb_dim: Dimension of word embeddings to be created
    """
    old_store = h5py.File(old_fpath, "r")
    store = h5py.File(str(old_fpath)[:-3] + "_fixed.h5", "a")
    old_fpath = str(old_fpath)
    # Setup data storage

    old_len = old_store["idx"].shape[0]
    print(f"old length idx: {old_store['idx'].shape[0]}")
    print(f"old length date: {old_store['date'].shape[0]}")
    print(f"old length emb: {old_store['emb_tensor'].shape[0]}")
    print(f"old length name: {old_store['name'].shape[0]}")
    print(f"old length label: {old_store['label'].shape[0]}")
    print(f"old length length: {old_store['length'].shape[0]}")

    pos = 0
    while pos + step_size < old_len:
        print(f"pos: {pos}")
        print(f"old_len: {old_len}")
        rows = get_rows(old_store, start=pos, step_size=step_size)

        idx_arr = rows["idx"]
        remove_index = []
        for i, idx in enumerate(idx_arr):
            if idx in idx_to_be_removed:
                print(f"element with idx {idx} should be popped!!!!\nIndex in batch is {i}")
                remove_index.append(i)
        

        for key, col_arr in rows.items():
            popped = 0 # Keep track of how many popped so that we don't overshoot after popping x elements
            new_col = col_arr.tolist()
            for idx in remove_index:
                new_col.pop(idx - popped)
                print(f"popped element at {idx-popped} in batch")
                popped += 1
            rows[key] = np.array(new_col)
            col_arr = None

        
        if pos == 0:
            first_time_setup_dataset(store, rows)

        else:   
            resize_and_append_datasets(store, rows)
        
        print(f"I am on pos {pos} and emb ds has shape: {store['idx'].shape}")
        rows = []
        pos += step_size

    if pos < old_len:
        remaining_rows = old_len-pos
        rows = get_rows(old_store, start=pos, step_size=remaining_rows)
        
        idx_arr = rows["idx"]
        remove_index = []
        for i, idx in enumerate(idx_arr):
            if idx in idx_to_be_removed:
                print(f"element with idx {idx} should be popped!!!!\nIndex in batch is {i}")
                remove_index.append(i)
        
        for key, col_arr in rows.items():
            new_col = col_arr.tolist()
            popped = 0
            for idx in remove_index:
                new_col.pop(idx - popped)
                print(f"popped element at {idx-popped} in batch")
                popped += 1
            rows[key] = np.array(new_col)
            col_arr = None

        if pos == 0:
            first_time_setup_dataset(store, rows)

        else:
            resize_and_append_datasets(store, rows)
        
        rows = [] # Hacky way to free up memory :)
        print(f"I am on pos {pos} and 'data' chunk has shape: {store['idx'].shape}")
    
    store.close()
    old_store.close()

if __name__ == "__main__":
    base_path = Path(os.path.abspath(__file__)).parents[1] / "features" / "embeddings" / "new" / "fix_row_embs"
    print(f"base_path: {base_path}")

    all_files = base_path.rglob("*.h5")
    print(f"all_files: {all_files}")

    files_512 = []
    files_256 = []
    for f in base_path.rglob("*.h5"):
        split_str = str(f).split("_")
        print(split_str)
        if "512.h5" in split_str: files_512.append(f)
        elif "256.h5" in split_str: files_256.append(f)
    

    print(f"files_512\n{files_512}")
    print(f"files_256\n{files_256}")

    for f in files_512:
        fix_embeddings(f, [3565])

    for f in files_256:
        fix_embeddings(f, [6982])