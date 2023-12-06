from gensim.models import Word2Vec
import gensim.downloader as api
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import csv
nltk.download('punkt')

#Train word2vec model
def train(df,model_fp):
    sentences = df['clue'].tolist()
    dataset = [word_tokenize(sentence.lower()) for sentence in sentences]
    m = Word2Vec(sentences=dataset, vector_size=100, window=5, min_count=1, workers=4)
    m.save(f"{model_fp}.model")

df = pd.read_csv("train_noblanks.csv")
df = df.astype(str)
model_fp = "word2vec"
train(df,model_fp)