import pytest
from llm_coder.parsers import PromptExampleParser

@pytest.fixture
def parser():
    return PromptExampleParser()

def test_parse_with_examples(parser):
    result = ">>> input1\n    output1\n>>> input2\n    output2"
    expected = [
        {"input": "input1", "output": "output1"},
        {"input": "input2", "output": "output2"}
    ]
    assert parser.parse(result) == expected

def test_parse_without_examples(parser):
    result = "Here is some text without examples."
    assert parser.parse(result) is None

def test_parse_empty_string(parser):
    result = ""
    assert parser.parse(result) is None