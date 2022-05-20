import sys

from numpy import isin

sys.path.append("../src/jobnet")
from preprocessing import rm_stops
import pytest
import pandas as pd
import datatest as dt


def test_rm_stops():
    da_stops = open("../stops/da_stops_lemmas.txt", "r")
    da_stops = da_stops.read().split()

    text = [
        "du",
        "har",
        "et",
        "solidt",
        "kendskab",
        "til",
        "branchen",
        "og",
        "forstaar",
        "vigtigheden",
        "i",
        "at",
        "holde",
        "sig",
        "relevant",
    ]
    no_stops = rm_stops(text, da_stops)

    assert isinstance(no_stops, list)

    assert no_stops == [
        "har",
        "et",
        "solidt",
        "kendskab",
        "branchen",
        "forstaar",
        "vigtigheden",
        "holde",
        "relevant",
    ]
