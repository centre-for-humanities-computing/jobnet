import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd


df = pd.read_pickle("../../data/pkl/dataset.pkl")

freq = Counter(lemma for post in df["nn_adj_lemmas"] for lemma in set(post))
df = pd.DataFrame(freq.items(), columns=["word", "frequency"]).sort_values(
    by="frequency", ascending=False
)

plt.rc("xtick", labelsize=25)
plt.rc("ytick", labelsize=25)

fig, axes = plt.subplots(figsize=(25, 15))
fig.subplots_adjust(bottom=0.15, left=0.2)

ax = sns.barplot(x="frequency", y="word", palette="Blues_r", data=df.head(30))

ax.xaxis.get_label().set_fontsize(25)
ax.yaxis.get_label().set_fontsize(25)
ax.axes.set_title("Most frequent words", fontweight="bold", size=40, y=1.03)

ax.tick_params(axis="x", colors="grey")
ax.tick_params(axis="y", colors="grey")

plt.xticks(fontsize=40)
plt.yticks(fontsize=25)

ax.set(xlabel="", ylabel="")

plt.savefig("../../figs/count.pdf")
