import sys

sys.path.append("../../data")
sys.path.append("../jobnet")
import json
import spacy
import pandas as pd
from preprocessing import (
    substitute_letter,
    clean_text,
    remove_html_commands,
    collect_nn_adj,
    rm_stops,
)
from polyglot.detect import Detector
from load import get_files

nlp = spacy.load("da_core_news_lg")

da_stops = open("../../stops/da_stops_lemmas.txt", "r")
da_stops = da_stops.read().split()

path = "../../data/jobnet-sample/"
files_paths = get_files(path)

descriptions = []

for file in files_paths:
    with open(file, encoding="utf-8") as f:
        data = json.load(f)
        descriptions.append(data["FormattedPurpose"])

cleaned_posts = [remove_html_commands(post) for post in descriptions]
cleaned_posts = [substitute_letter(post) for post in cleaned_posts]
cleaned_posts = [clean_text(post) for post in cleaned_posts]
cleaned_posts = [post for post in cleaned_posts if post != ""]

da_posts = []

for post in cleaned_posts:
    detector = Detector(post)
    if detector.language.code == "da" and detector.language.confidence >= 90:
        da_posts.append(post)


tags = ["PROPN", "NOUN", "ADJ"]

nn_adj_lemmas = []

for post in da_posts:
    lemmas = collect_nn_adj(post, nlp, tags)
    nn_adj_lemmas.append(lemmas)

no_stops = [rm_stops(post, da_stops) for post in nn_adj_lemmas]


df = pd.DataFrame(columns=["cleaned_description", "nn_adj_lemmas"])
df["cleaned_description"] = da_posts
df["nn_adj_lemmas"] = no_stops
df.to_pickle("../../data/pkl/dataset.pkl")
