"""Pipeline for nodes graph creation"""
import pandas as pd
from nltk.util import bigrams
import itertools
import collections
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from collections import Counter


df = pd.read_pickle("../../data/pkl/dataset.pkl")

lemmas = df["nn_adj_lemmas"]

terms_bigram = [list(bigrams(post)) for post in lemmas]
bigram = list(itertools.chain(*terms_bigram))
bigram_counts = collections.Counter(bigram)
bigrams_df = pd.DataFrame(bigram_counts.most_common(30), columns=["bigram", "count"])

word_freq = Counter(lemma for post in df["nn_adj_lemmas"] for lemma in set(post))
w_freq_df = pd.DataFrame(word_freq.items(), columns=["word", "frequency"]).sort_values(
    by="frequency", ascending=False
)

# Blues
# palette_edges = ["#f8fcfd", "#e9f5f8", "#cbe6ef", "#bcdfeb", "#62b4cf", "#0000FF", "#0047AB", "#00008b"]

# Greens
palette_nodes = ["#8fbc8f", "#3cb371", "#2e8b57", "#006400"]

# Create dictionary of bigrams and their freq-cy
d = bigrams_df.set_index("bigram").T.to_dict("records")

# Create network plot
G = nx.Graph()

# Create connections between nodes based on bigrams frequency
for key, value in d[0].items():
    G.add_edge(key[0], key[1], weight=(value))

fig, ax = plt.subplots(figsize=(12, 10))
pos = nx.spring_layout(G, k=4)

# Nodes scaled by lemmas freq present in bigrams
d = w_freq_df.to_dict(orient="split")["data"]
d = [(int(word[1])) * 2 for node in G.nodes() for word in d if word[0] == node]

nx.draw_networkx(
    G,
    pos,
    font_size=10,
    width=3,  # edges width (can be scaled)
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
        bbox=dict(facecolor="#FFF0F5", alpha=0.5, edgecolor = "grey", pad=3.5),
        horizontalalignment="center",
    #    color = '#696969',
        weight='bold',
        fontsize=6,
    )

fig.patch.set_visible(False)
ax.axis("off")
plt.savefig("../../figs/bigram_nn_adj_graph.pdf")
