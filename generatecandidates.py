from gensim.models import Word2Vec
import gensim.downloader as api
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')

#Get topK clue answers for given model and clue
def topK(model,test_input,k=50):
    def sentence_vector(sentence, model):
        words = [word for word in sentence if word in model.wv]
        return np.mean(model.wv[words], axis=0) if words else np.zeros(model.vector_size)

    def process_batch(batch, model, test_vector):
        batch = batch.copy()
        batch['sentence_vector'] = batch['clue'].apply(lambda x: sentence_vector(x.split(), model))
        batch['similarity'] = batch['sentence_vector'].apply(lambda x: cosine_similarity([x], [test_vector])[0][0])
        return batch[['clue', 'answer', 'similarity']]

    test_vector = sentence_vector(test_input.lower().split(), model)
    batch_size = 10000
    results = []
    
    for start in range(0, len(df), batch_size):
        batch = df[start:start + batch_size]
        results.append(process_batch(batch, model, test_vector))
    
    final_results = pd.concat(results)
    topK = final_results.nlargest(k, 'similarity')
    
    topK_answers = topK['answer'].tolist()
    return topK_answers

df = pd.read_csv("train_noblanks.csv")
df = df.astype(str)
model_fp = "word2vec"
model = Word2Vec.load(f"{model_fp}.model")

#sample call
print(topK(model,"Carol's about to tolerate bad language",model))