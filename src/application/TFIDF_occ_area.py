"""Pipeline for TF-IDF weighed word frequency count per occupation area"""
import sys

sys.path.append("../jobnet")

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import identity_tokenizer


df = pd.read_pickle("../../data/pkl/dataset.pkl")

vectorizer = TfidfVectorizer(
    lowercase=False, tokenizer=identity_tokenizer, ngram_range=(1, 1)
)

occ_areas = list(df["occupation_area"].unique())

for area in occ_areas:

    df0 = df[df["occupation_area"] == area]
    lemmas = df0[str(sys.argv[1])]
    tfidf_vectors = vectorizer.fit_transform(lemmas)

    tfidf_weights = [
        (word, tfidf_vectors.getcol(idx).sum())
        for word, idx in vectorizer.vocabulary_.items()
    ]

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        random_state=1,
        background_color="white",
        colormap="inferno",
        collocations=False,
    ).fit_words(dict(tfidf_weights))

    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud)
    plt.axis("off")
    plot_name = f"../../figs/occ_area/TF_IDF/{area}_{str(sys.argv[1])}_TFIDF.pdf"
    plt.savefig(plot_name)
