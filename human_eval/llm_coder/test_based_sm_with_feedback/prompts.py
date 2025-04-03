FIRST_GENERATION_PROMPT = """Complete the following code. Use ```python to put the completed Python code, including the necessary imports, in markdown quotes:\n{problem}"""


ITERATIVE_PROMPT = """Complete the following code. Use ```python to put the completed Python code, including the necessary imports, in markdown quotes:
{problem}

## Current Attempt
There is already one attempt of the prompt:
```python
{existing_solution}
```

## Passed Test Cases
The current solution passed the following test cases:
{success_test_cases}

## Issues
However, the current solution has some issues with the following test cases.
{failed_test_cases}

Let's do it step by step. Improve the current solution to pass the failed test cases while keeping the existing solution intact.
"""

AGENT_DESCRIPTION_PROMPT = """You are a Python programming expert tasked with solving algorithmic problems. Your task is to choose the best next action whether to generate a solution or test cases for the programming problem.
"""

TEST_GENERATION_PROMPT = """You are a Python programming expert tasked with solving algorithmic problems. Your task is given the following problem description, write a function with test cases to test the solution program for this problem using the input space partitioning strategy.

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
    assert candidate(1, 2) == 3, "Test positive integers failed: for input (1, 2), expected output 3"
    # negative integers
    assert candidate(-1, -2) == -3, "Test negative integers failed: for input (-1, -2), expected output -3"
    # zero values
    assert candidate(0, 5) == 5, "Test zero values failed: for input (0, 5), expected output 5"
    # mixed signs
    assert candidate(7, -2) == 5, "Test mixed signs failed: for input (7, -2), expected output 5"
```

Make sure the input satisfy the input constraints and do not include any invalid inputs.
Make sure to consider the precision issues when comparing floating-point numbers.
Follow the above example, generate at most 5 test cases in a Markdown code block.

Input: {problem}
Output:
"""
