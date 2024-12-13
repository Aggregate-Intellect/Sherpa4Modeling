FIRST_GENERATION_PROMPT = """You are a Python programming expert tasked with solving algorithmic problems. Your task is to write a Python function that satisfies the following requirements. Please ensure that your solution:
1. Is written in Python 3.
2. Is aligned with the problem description written in the function comment.
2. Passes all the test cases provided in the problem description.
3. Give a runnable problem including potential imports

First explain how to solve the programming challenge, then output the final Python program as a Markdown code block:
```python
```

## Problem
{problem}

## Solution
"""


ITERATIVE_PROMPT = """You are a Python programming expert tasked with solving algorithmic problems. Your task is to write a Python function that satisfies the following requirements. Please ensure that your solution:
1. Is written in Python 3.
2. Is aligned with the problem description written in the function comment.
2. Passes all the test cases provided in the problem description.
3. Give a runnable problem including potential imports

First explain how to solve the programming challenge, then output the final Python program as a Markdown code block:
```python
```

## Problem
{problem}

## Current Attempt
There is already one attempt of the prompt:
```pythoh
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
