"""Pipeline for word frequency count"""
import sys

sys.path.append("../jobnet")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
from preprocessing import create_dfs_list


df = pd.read_pickle("../../data/pkl/dataset.pkl")
occ_areas = list(df["occupation_area"].unique())

dfs_series = create_dfs_list(df, str(sys.argv[1]), occ_areas)

for df in dfs_series:

    name = df.name

    freq = Counter(lemma for post in df for lemma in set(post))
    df_freq = pd.DataFrame(freq.items(), columns=["word", "frequency"]).sort_values(
        by="frequency", ascending=False
    )

    plt.rc("xtick", labelsize=25)
    plt.rc("ytick", labelsize=25)

    fig, axes = plt.subplots(figsize=(25, 15))
    fig.subplots_adjust(bottom=0.15, left=0.2)

    ax = sns.barplot(x="frequency", y="word", palette="Greens_r", data=df_freq.head(30))

    ax.xaxis.get_label().set_fontsize(25)
    ax.yaxis.get_label().set_fontsize(25)
    ax.axes.set_title("Most frequent words", fontweight="bold", size=40, y=1.03)

    ax.tick_params(axis="x", colors="grey")
    ax.tick_params(axis="y", colors="grey")

    plt.xticks(fontsize=40)
    plt.yticks(fontsize=25)

    ax.set(xlabel="", ylabel="")

    plot_name = f"../../figs/word_count/{name}_unigrams_{str(sys.argv[1])}.pdf"
    plt.savefig(plot_name)
    plt.close()
