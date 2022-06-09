"""Pipeline for nodes graphs creation"""
import sys

sys.path.append("../jobnet")
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from collections import Counter
import pandas as pd
from nltk.util import bigrams
import itertools
import networkx as nx
from preprocessing import create_dfs_list


df = pd.read_pickle("../../data/pkl/dataset.pkl")
occ_areas = list(df["occupation_area"].unique())

dfs_series = create_dfs_list(df, str(sys.argv[1]), occ_areas)

for df in dfs_series:

    name = df.name

    terms_bigram = [list(bigrams(post)) for post in df]
    bigram = list(itertools.chain(*terms_bigram))
    bigram_counts = collections.Counter(bigram)
    bigrams_df = pd.DataFrame(
        bigram_counts.most_common(30), columns=["bigram", "count"]
    )

    word_freq = Counter(lemma for post in df for lemma in set(post))
    w_freq_df = pd.DataFrame(
        word_freq.items(), columns=["word", "frequency"]
    ).sort_values(by="frequency", ascending=False)

    # Greens
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
    plot_name = f"../../figs/bigrams/{name}_bigrams_{str(sys.argv[1])}.pdf"
    plt.savefig(plot_name)
    plt.close()
