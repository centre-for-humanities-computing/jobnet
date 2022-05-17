"""Pipeline for preparing a list of Danish stop words"""
import spacy
import sys

sys.path.append("../jobnet")
from preprocessing import substitute_letter

da_stops = open("../../stops/da_stop_words.txt", "r")
da_stops = da_stops.read().split()


nlp = spacy.load("da_core_news_lg")

da_lemmas = []

for stop in da_stops:
    doc = nlp(stop)
    for token in doc:
        da_lemmas.append(token.lemma_.lower())

da_lemmas_sub = [substitute_letter(post) for post in da_lemmas]

da_stops = da_lemmas + da_lemmas_sub

f = open("../../stops/da_stops_lemmas.txt", "w")

for lemma in set(da_stops):
    f.write(lemma)
    f.write("\n")
f.close()
