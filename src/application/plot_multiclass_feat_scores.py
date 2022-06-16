import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_features = pd.read_pickle("../../data/pkl/multiclass_feature_analysis.pkl")

sectors = ["akademisk", "pædagogisk", "salg", "sundhed"]

for sector in sectors:

    largest = df_features.nlargest(30, [sector])
    largest = largest.reset_index(drop=True)

    smallest = df_features.nsmallest(30, [sector])
    smallest = smallest.reset_index(drop=True)

    plt.rc("xtick", labelsize=25)
    plt.rc("ytick", labelsize=25)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 11))
    fig.suptitle("Feature Importance Scores", fontweight="bold", size=25, y=0.95)
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )

    sns.barplot(ax=ax1, x=sector, y="word", palette="Greens_r", data=largest)
    sns.barplot(ax=ax2, x=sector, y="word", palette="Purples_r", data=smallest)

    ax1.axes.set_title(f"Largest scores", fontweight="bold", size=18, y=1)
    ax1.tick_params(axis="x", colors="grey", labelsize=12)
    ax1.tick_params(axis="y", colors="grey", labelsize=12)

    ax2.axes.set_title("Smallest scores", fontweight="bold", size=18, y=1)
    ax2.tick_params(axis="x", colors="grey", labelsize=12)
    ax2.tick_params(axis="y", colors="grey", labelsize=12)

    ax1.set(xlabel="", ylabel="")
    ax2.set(xlabel="", ylabel="")
    plt.savefig(f"../../figs/feature_importance/multiclass_classifier_{sector}.pdf")
