FIRST_GENERATION_PROMPT = """Complete the following code. Use ```python to put the completed Python code, including the necessary imports, in markdown quotes:\n{problem}"""


ITERATIVE_PROMPT = """Complete the following code. Use ```python to put the completed Python code, including the necessary imports, in markdown quotes:
{problem}

## Current Attempt
There is already one attempt of the prompt:
```python
{existing_solution}
```

## Issues
However, the current solution has some issues and failed some of the following test cases.
{test_cases}

Improve the current solution to pass all the test cases.
## Solution
"""

AGENT_DESCRIPTION_PROMPT = """You are a Python programming expert tasked with solving algorithmic problems. Your task is to choose the best next action whether to generate a solution or test cases for the programming problem.
"""

TEST_GENERATION_PROMPT = """You are a Python programming expert tasked with solving algorithmic problems. Your task is given the following problem description, write a function with test cases to test the solution program for this problem.

For example:
Input: 
def sum(a: int, b: int):
    \"\"\" Return the sume of two integers 
    >>> sum(1, 2)
    3
    >>> sum(0, 0)
    0
    \"\"\"
Output:
```python
def check(candidate):
    # positive integers
    assert candidate(1, 2) == 3
    # negative integers
    assert candidate(-1, -2) == -3
    # zero values
    assert candidate(0, 5) == 5
    # mixed signs
    assert candidate(7, -2) == 5
```

Make sure the input satisfy the input constraints and do not include any invalid inputs.
Make sure to consider the precision issues when comparing floating-point numbers.
Follow the above example, generate at most 5 test cases in a Markdown code block.

Input: {problem}
Output:
"""
