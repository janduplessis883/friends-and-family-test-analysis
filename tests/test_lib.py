from friendsfamilytest.data import *
from friendsfamilytest.utils import *
import re

def test_replace_surname():
    text = "My name is burhan adib is my friend"
    expected_output = "My name is b a is my friend"
    assert replace_surname(text) == expected_output

    text = "lula and lula went to joyce s house"
    expected_output = "l and l went to j s house"
    assert replace_surname(text) == expected_output

    text = "christine jan and orietta are colleagues"
    expected_output = "c j and o are colleagues"
    assert replace_surname(text) == expected_output