import sys

sys.path.append("../src/jobnet")
from preprocessing import collect_tokens
import pytest
import pandas as pd
import datatest as dt
import spacy


def test_collect_tokens():
    nlp = spacy.load("da_core_news_lg")
    text = "i baren copenhagen go hoteltilbyder vaerelser til budgetvenlige priser"
    tokens = collect_tokens(text, nlp)

    assert isinstance(tokens, list)
    assert tokens == [
        "i",
        "baren",
        "copenhagen",
        "go",
        "hoteltilbyder",
        "vaerelser",
        "til",
        "budgetvenlige",
        "priser",
    ]


