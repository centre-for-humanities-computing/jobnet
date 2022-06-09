"""Pipeline for TF-IDF weighed word frequency count"""
import sys

sys.path.append("../../data")
sys.path.append("../jobnet")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import identity_tokenizer, create_dfs_list


df = pd.read_pickle("../../data/pkl/dataset.pkl")
occ_areas = list(df["occupation_area"].unique())

dfs_series = create_dfs_list(df, str(sys.argv[1]), occ_areas)

for df in dfs_series:

    name = df.name

    vectorizer = TfidfVectorizer(
        lowercase=False, tokenizer=identity_tokenizer, ngram_range=(1, 1)
    )
    tfidf_vectors = vectorizer.fit_transform(df)

    tfidf_weights = [
        (word, tfidf_vectors.getcol(idx).sum())
        for word, idx in vectorizer.vocabulary_.items()
    ]

    tf_df = pd.DataFrame(tfidf_weights, columns=["word", "frequency"])
    tf_df = tf_df.sort_values(by="frequency", ascending=False)

    plt.rc("xtick", labelsize=25)
    plt.rc("ytick", labelsize=25)

    fig, axes = plt.subplots(figsize=(25, 15))
    fig.subplots_adjust(bottom=0.15, left=0.2)

    ax = sns.barplot(x="frequency", y="word", palette="Blues_r", data=tf_df.head(30))

    ax.xaxis.get_label().set_fontsize(25)
    ax.yaxis.get_label().set_fontsize(25)
    ax.axes.set_title("Most frequent words", fontweight="bold", size=40, y=1.03)

    ax.tick_params(axis="x", colors="grey")
    ax.tick_params(axis="y", colors="grey")

    plt.xticks(fontsize=40)
    plt.yticks(fontsize=25)

    ax.set(xlabel="", ylabel="")

    plot_name = f"../../figs/TFIDF/{name}__TFIDF_{str(sys.argv[1])}.pdf"
    plt.savefig(plot_name)
    plt.close()
