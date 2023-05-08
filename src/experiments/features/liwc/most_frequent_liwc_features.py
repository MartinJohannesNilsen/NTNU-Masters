import pandas as pd
import sys
import os
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(str(Path(os.path.abspath(__file__)).parents[3] / "utils"))
from utils.word_embeddings import _tokenize_with_preprocessing
from utils.word_emb_utils import create_vocab_w_idx



liwc_path = Path(os.path.abspath(__file__)).parents[0] / "2022"

train_df = pd.read_csv(str(liwc_path / "train_sliced_stair_twitter.csv"))
test_df = pd.read_csv(str(liwc_path / "test_sliced_stair_twitter.csv"))
val_df = pd.read_csv(str(liwc_path / "shooter_hold_out_test.csv"))

all_df = pd.concat([train_df, test_df, val_df], axis=0)
all_df["text"] = all_df["text"].map(lambda a: _tokenize_with_preprocessing(a))
all_df["text"] = all_df["text"].map(lambda a: _tokenize_with_preprocessing(a))

vocab = create_vocab_w_idx(all_df)

def check_in_vocab(df, vocab):
    # Check if word is in vocab, drop if not
    text = df["text"]
    new_tokens = []
    for word in text:
        if word in vocab:
            new_tokens.append(word)

    return new_tokens

all_df["text"] = all_df["text"].map(lambda a: check_in_vocab(a))

lemmatizer = WordNetLemmatizer()
all_df["text"] = all_df["text"].map(lambda a: lemmatizer.lemmatize(a))
all_df["text"] = all_df["text"].map(lambda a: " ".join(a))


shooter_df = all_df[all_df["label"] == 1]
non_shooter_df = all_df[all_df["label"] == 0]

shooter_arr = shooter_df["text"].values.tolist()
non_shooter_arr = non_shooter_df["text"].values.tolist()

shooter_str = " ".join(shooter_arr)
non_shooter_str = " ".join(non_shooter_arr)

shooter_vectorizer = TfidfVectorizer(stop_words="english")
non_shooter_vectorizer = TfidfVectorizer(stop_words="english")

shooter_vectorized = shooter_vectorizer.fit_transform(shooter_str)
non_shooter_vectorized = non_shooter_vectorizer.fit_transform(non_shooter_str)

shooter_feature_names = shooter_vectorizer.get_feature_names()
non_shooter_feature_names = non_shooter_vectorizer.get_feature_names()

shooter_dense = shooter_vectorized.todense()
non_shooter_dense = non_shooter_vectorized.todense()

lst1 = shooter_dense.tolist()
lst2 = non_shooter_dense.tolist()

shooter_occurences = pd.DataFrame(lst1, columns=shooter_feature_names)
non_shooter_occurences = pd.DataFrame(lst2, columns=non_shooter_feature_names)

print(shooter_occurences)




