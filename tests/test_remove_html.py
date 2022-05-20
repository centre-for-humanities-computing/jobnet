import sys

sys.path.append("../src/jobnet")
from preprocessing import remove_html_commands
import pytest
import pandas as pd
import datatest as dt


def test_remove_html_commands():
    sample = "<p>I Lissabon er der s&oslash;rget for billig skat, sol og bolig. Vi s&oslash;rger ogs&aring; for fly hertil samt afhentning i lufthavnen. </p>"
    cleaned = remove_html_commands(sample)

    assert isinstance(cleaned, str)
    assert (
        cleaned
        == "I Lissabon er der sørget for billig skat, sol og bolig. Vi sørger også for fly hertil samt afhentning i lufthavnen."
    )
