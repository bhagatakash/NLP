import numpy as np
import pandas as pd
import nltk
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import pickle

nltk.download('punkt') # one time execution
import re

# Extract word vectors
pkl_file = open('vectors.pkl', 'rb')
word_embeddings = pickle.load(pkl_file)
pkl_file.close()
#Ref - https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/



def short_desc(text, word_embeddings, lines=10):
    sentences=sent_tokenize(text)
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    # function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new
    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    sentence_vectors = []
    for i in clean_sentences:
      if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i.split()])/(len(i.split())+0.001)
      else:
        v = np.zeros((300,))
      sentence_vectors.append(v)
    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
      for j in range(len(sentences)):
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    # Extract top lines sentences as the summary
    x=''
    for i in range(lines):
      x=x+' '+ranked_sentences[i][1]
    x=x.replace('\n',' ')
    return x
    

