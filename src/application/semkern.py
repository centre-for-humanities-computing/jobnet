"""Pipeline for training word2vec model on job posts and plotting semantic graphs"""
import sys

sys.path.append("../../../semantisk-kernel/semkern")
import pandas as pd
from gensim.models import Word2Vec
from model import train_model
from graphing import plot


df = pd.read_pickle("../../data/pkl/dataset.pkl")

posts = df["lemmas"].to_list()

model = train_model(posts)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")

seeds = ["professor"]

k = 5
m = 3

fig = plot(seeds, k, m, model)
fig.show()
fig.write_image("../../figs/w2v/professor.pdf")
