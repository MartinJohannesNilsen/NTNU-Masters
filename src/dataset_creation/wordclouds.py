import pandas as pd
import sys
import os
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from experiments.utils.word_embeddings import _tokenize_with_preprocessing
from experiments.utils.word_emb_utils import create_vocab_w_idx, tokenize_with_preprocessing_drop_len
from csv import QUOTE_NONE
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


data_path = Path(os.path.abspath(__file__)).parents[0] / "data" / "train_test" / "new"

train_df = pd.read_csv(str(data_path / "train_sliced_stair_twitter_512.csv"), sep="‎", quoting=QUOTE_NONE, engine="python")
test_df = pd.read_csv(str(data_path / "test_sliced_stair_twitter_512.csv"), sep="‎", quoting=QUOTE_NONE, engine="python")
val_df = pd.read_csv(str(data_path / "val_sliced_stair_twitter_512.csv"), sep="‎", quoting=QUOTE_NONE, engine="python")

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text, lemmatizer):
    new_text = []
    for word in text:
        new_text.append(lemmatizer.lemmatize(word))

    return new_text

all_df = pd.concat([train_df, test_df, val_df], axis=0)
all_df["text"] = all_df["text"].map(lambda a: tokenize_with_preprocessing_drop_len(a))
all_df["text"] = all_df["text"].map(lambda a: " ".join(lemmatize_text(a, lemmatizer)))
#all_df["text"] = all_df["text"].map(lambda a: a.split(" "))
#all_df["text"] = all_df["text"].map(lambda a: _tokenize_with_preprocessing(a))

shooter_df = all_df[all_df["label"] == 1]
non_shooter_df = all_df[all_df["label"] == 0]
 
stopwords = set(STOPWORDS)
 
def get_text_str(df):
    words = ""
    for text in df["text"].astype(str):       
        words += text
    
    return words
 

""" shooter_wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(get_text_str(shooter_df)) """

non_shooter_wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(get_text_str(non_shooter_df))
 
# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(non_shooter_wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()




""" all_df["text"] = all_df["text"].map(lambda a: lemmatize_text(a, lemmatizer))


print(all_df)
#all_df["text"] = all_df["text"].map(lambda a: " ".join(a))
#all_df["text"] = all_df["text"].map(lambda a: " ".join(a))


shooter_df = all_df[all_df["label"] == 1]
non_shooter_df = all_df[all_df["label"] == 0]

def get_text(text_df):
    new_text = ""
    for text in text_df["text"]:
        new_text += f" {' '.join(text).strip()} "
    return new_text

shooter_str = get_text(shooter_df)
non_shooter_str = get_text(non_shooter_df)

shooter_vectorizer = TfidfVectorizer(stop_words="english")
non_shooter_vectorizer = TfidfVectorizer(stop_words="english")

shooter_vectorized = shooter_vectorizer.fit_transform(shooter_str)
non_shooter_vectorized = non_shooter_vectorizer.fit_transform(non_shooter_str)
shooter_feature_names = shooter_vectorizer.get_feature_names_out()
non_shooter_feature_names = non_shooter_vectorizer.get_feature_names_out()

shooter_dense = shooter_vectorized.todense()
non_shooter_dense = non_shooter_vectorized.todense()

lst1 = shooter_dense.tolist()
lst2 = non_shooter_dense.tolist()

shooter_occurences = pd.DataFrame(lst1, columns=shooter_feature_names)
non_shooter_occurences = pd.DataFrame(lst2, columns=non_shooter_feature_names)

print(shooter_occurences)
 """



