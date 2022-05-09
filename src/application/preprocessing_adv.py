import sys

sys.path.append("../data")
sys.path.append("../jobnet")
import json
import spacy
from preprocessing import (
    substitute_letter,
    clean_text,
    remove_html_commands,
    collect_lemmas,
    rm_stops,
)
from polyglot.detect import Detector
from load import get_files


da_stops = open("../../stops/da_stops_lemmas.txt", "r")
da_stops = da_stops.read().split()

en_stops = open("../../stops/en_stops_lemmas.txt", "r")
en_stops = en_stops.read().split()

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
en_posts = []
other = []

for post in cleaned_posts:
    detector = Detector(post)
    if detector.language.code == "da":
        da_posts.append(post)
    elif detector.language.code == "en":
        en_posts.append(post)
    else:
        other.append(post)

nlp = spacy.load("da_core_news_lg")
da_lemmas = [collect_lemmas(post, nlp) for post in da_posts]

nlp = spacy.load("en_core_web_lg")
en_lemmas = [collect_lemmas(post, nlp) for post in en_posts]

da_lemmas = [rm_stops(post, da_stops) for post in da_lemmas]
en_lemmas = [rm_stops(post, en_stops) for post in en_lemmas]
