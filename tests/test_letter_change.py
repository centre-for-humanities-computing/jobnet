import sys

sys.path.append("../src/jobnet")
from preprocessing import substitute_letter
import pytest
import pandas as pd
import datatest as dt


def test_substitute_letter():

    sample0 = "Vi sørger også for fly hertil samt afhentning i lufthavnen."
    sample1 = "Jeg bor i Århus Ø"
    sample3 = "forretningsrejsende og stamgæster"
    sample4 = "Æg"

    sample0_changed = substitute_letter(sample0)
    sample1_changed = substitute_letter(sample1)
    sample3_changed = substitute_letter(sample3)
    sample4_changed = substitute_letter(sample4)

    assert isinstance(sample0_changed, str)
    assert (
        sample0_changed
        == "Vi soerger ogsaa for fly hertil samt afhentning i lufthavnen."
    )

    assert isinstance(sample1_changed, str)
    assert sample1_changed == "Jeg bor i aarhus oe"

    assert isinstance(sample3_changed, str)
    assert sample3_changed == "forretningsrejsende og stamgaester"

    assert isinstance(sample4_changed, str)
    assert sample4_changed == "aeg"
