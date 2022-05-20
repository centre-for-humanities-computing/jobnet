import sys

sys.path.append("../src/jobnet")
from preprocessing import collect_nn_adj
import pytest
import pandas as pd
import datatest as dt
import spacy


def test_collect_lemmas():
    nlp = spacy.load("da_core_news_lg")
    text = "i baren copenhagen go hoteltilbyder vaerelser til budgetvenlige priser"
    pos_tags = ["PROPN", "NOUN", "ADJ"]
    nn_adj = collect_nn_adj(text, nlp, pos_tags)

    assert isinstance(nn_adj, list)
    assert nn_adj == [
        "bar",
        "copenhagen",
        "hoteltilbyder",
        "vaerelser",
        "budgetvenlige",
        "pris",
    ]
