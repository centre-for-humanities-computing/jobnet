import sys

sys.path.append("../src/jobnet")
from preprocessing import clean_text
import pytest
import pandas as pd
import datatest as dt


def test_clean_text():

    sample = "Here you can find the   definiton for the dot   product, https://en.wikipedia.org/wiki/Dot_product 1111."
    cleaned = clean_text(sample)

    assert isinstance(cleaned, str)
    assert cleaned == "here you can find the definiton for the dot product"
