"""Pipeline for nodes graph creation per occupation area"""
import sys

sns.set()
import pandas as pd
from nltk.util import bigrams
import itertools
import collections
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


df = pd.read_pickle("../../data/pkl/dataset.pkl")
occ_areas = list(df["occupation_area"].unique())


for area in occ_areas:

    df0 = df[df["occupation_area"] == area]
    lemmas = df0[str(sys.argv[1])]

    terms_bigram = [list(bigrams(post)) for post in lemmas]
    bigram = list(itertools.chain(*terms_bigram))
    bigram_counts = collections.Counter(bigram)
    bigrams_df = pd.DataFrame(
        bigram_counts.most_common(30), columns=["bigram", "count"]
    )

    word_freq = Counter(lemma for post in df[str(sys.argv[1])] for lemma in set(post))
    w_freq_df = pd.DataFrame(
        word_freq.items(), columns=["word", "frequency"]
    ).sort_values(by="frequency", ascending=False)

    palette_nodes = ["#8fbc8f", "#3cb371", "#2e8b57", "#006400"]

    d = bigrams_df.set_index("bigram").T.to_dict("records")

    G = nx.Graph()

    for key, value in d[0].items():
        G.add_edge(key[0], key[1], weight=(value))

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=4)

    d = w_freq_df.to_dict(orient="split")["data"]
    d = [(int(word[1])) * 2 for node in G.nodes() for word in d if word[0] == node]

    nx.draw_networkx(
        G,
        pos,
        font_size=10,
        width=3,
        cmap="Blues",
        edge_color=palette_nodes,
        node_color=d,
        with_labels=False,
        ax=ax,
    )

    for key, value in pos.items():
        x, y = value[0], value[1]
        ax.text(
            x,
            y,
            s=key,
            bbox=dict(facecolor="#FFF0F5", alpha=0.5, edgecolor="grey", pad=3.5),
            horizontalalignment="center",
            weight="bold",
            fontsize=6,
        )

    fig.patch.set_visible(False)
    ax.axis("off")
    plot_name = f"../../figs/occ_area/bigrams/{area}_bigram_{str(sys.argv[1])}.pdf"
    plt.savefig(plot_name)
