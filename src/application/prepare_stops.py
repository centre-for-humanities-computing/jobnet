import spacy
import sys

sys.path.append("../jobnet")
from preprocessing import substitute_letter

da_stops = open("../../stops/da_stop_words.txt", "r")
da_stops = da_stops.read().split()

en_stops = open("../../stops/en_stop_words.txt", "r")
en_stops = en_stops.read().split()

nlp = spacy.load("en_core_web_lg")

en_lemmas = []

for stop in en_stops:
    doc = nlp(stop)
    for token in doc:
        en_lemmas.append(token.lemma_.lower())

f = open("../../stops/en_stops_lemmas.txt", "w")

for lemma in set(en_lemmas):
    f.write(lemma)
    f.write("\n")
f.close()

nlp = spacy.load("da_core_news_lg")

da_lemmas = []

for stop in da_stops:
    doc = nlp(stop)
    for token in doc:
        da_lemmas.append(token.lemma_.lower())

da_lemmas = [substitute_letter(post) for post in da_lemmas]

f = open("../../stops/da_stops_lemmas.txt", "w")

for lemma in set(da_lemmas):
    f.write(lemma)
    f.write("\n")
f.close()
