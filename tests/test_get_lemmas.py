import sys

sys.path.append("../src/jobnet")
from preprocessing import collect_lemmas
import pytest
import pandas as pd
import datatest as dt
import spacy


def test_collect_lemmas():
    nlp = spacy.load("da_core_news_lg")
    text = "i baren copenhagen go hoteltilbyder vaerelser til budgetvenlige priser"
    lemmas = collect_lemmas(text, nlp)

    assert isinstance(lemmas, list)
    assert lemmas == [
        "i",
        "bar",
        "copenhagen",
        "go",
        "hoteltilbyder",
        "vaerelser",
        "til",
        "budgetvenlige",
        "pris",
    ]
