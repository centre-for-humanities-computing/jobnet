"""Pipeline for collection and processing of job posts"""
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
    collect_tokens,
    collect_lemmas,
    collect_nn_adj,
    rm_stops,
)
from polyglot.detect import Detector
from load import get_files

nlp = spacy.load("da_core_news_lg")

da_stops = open("../../stops/da_stops_lemmas.txt", "r")
da_stops = da_stops.read().split()

root_dir = "../../data/jobnet/"
files_paths = get_files(root_dir)

descriptions = []
occupation_area = []
occupation_group = []
occupation = []


for file in files_paths:
    with open(file, encoding="utf-8") as f:
        data = json.load(f)
        try:
            if data["isExternal"] == False:
                descriptions.append(data["details"]["FormattedPurpose"])
                occupation_area.append(data["summary"]["OccupationArea"])
                occupation_group.append(data["summary"]["OccupationGroup"])
                occupation.append(data["summary"]["Occupation"])
            else:
                continue
        except TypeError:
            continue

cleaned_posts = [remove_html_commands(post) for post in descriptions]
cleaned_posts = [substitute_letter(post) for post in cleaned_posts]
cleaned_posts = [clean_text(post) for post in cleaned_posts]

df = pd.DataFrame(
    columns=[
        "occupation_area",
        "occupation_group",
        "occupation",
        "cleaned_description",
        "da_description",
        "tokens",
        "lemmas",
        "nn_adj_lemmas",
    ]
)

df["occupation_area"] = occupation_area
df["occupation_group"] = occupation_group
df["occupation"] = occupation
df["cleaned_description"] = cleaned_posts

df = df.loc[df["cleaned_description"] != ""]

df = df.drop_duplicates(subset=['cleaned_description'], keep='first')

for row in range(len(df)):
    detector = Detector(df["cleaned_description"].iloc[row], quiet=True)
    if detector.language.code == "da" and detector.language.confidence >= 90:
        df["da_description"].iloc[row] = True
    else:
        df["da_description"].iloc[row] = False

df = df.loc[df["da_description"] == True]
df = df.drop(["da_description"], axis=1)

tags = ["PROPN", "NOUN", "ADJ"]

cleaned_descriptions = df["cleaned_description"].to_list()
tokens = [collect_tokens(post, nlp) for post in cleaned_descriptions]
all_lemmas = [collect_lemmas(post, nlp) for post in cleaned_descriptions]
nn_adj = [collect_nn_adj(post, nlp, tags) for post in cleaned_descriptions]
no_stops_tokens = [rm_stops(post, da_stops) for post in tokens]
no_stops_lemmas = [rm_stops(post, da_stops) for post in all_lemmas]
no_stops_nn_adj = [rm_stops(post, da_stops) for post in nn_adj]
df["tokens"] = no_stops_tokens
df["lemmas"] = no_stops_lemmas
df["nn_adj_lemmas"] = no_stops_nn_adj

df.to_pickle("../../data/pkl/dataset.pkl")
