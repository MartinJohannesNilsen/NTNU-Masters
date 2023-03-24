# Imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from sklearn import decomposition
from pathlib import Path
import matplotlib as plt
import pandas as pd
import numpy as np
import nltk
import re

# CONSTANTS
data_path = (Path(__file__).parents[2] / "data/")
cho_text_path = data_path / "school_shooters/Seung_Hui_Cho/data.csv"
sep = "‎"

# Load some data
df = pd.read_csv(cho_text_path, sep=sep)
print(df.head())

X_train, X_test = train_test_split(df, test_size=0.6, random_state=1)
x_train_counts = X_train["text"].value_counts()
x_test_counts = X_test["text"].value_counts()

stemmer = PorterStemmer()

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens

# LDA algorithm
tf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words="english", use_idf=False, norm=None, max_df=0.75)
tf_vectors = tf_vectorizer.fit_transform(X_train["text"])
print(tf_vectors.A)

#tokens = [word for word in nltk.word_tokenize(df)]
lda = decomposition.LatentDirichletAllocation(n_components=6, max_iter=3, learning_method="online", learning_offset=50)
w1 = lda.fit_transform(tf_vectors)
h1 = lda.components_