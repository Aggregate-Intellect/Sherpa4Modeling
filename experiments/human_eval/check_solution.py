from parsers import PromptExampleParser, PythonOutputParser
from human_eval.execution import check_correctness
import json
from human_eval.data import read_problems
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# TEST_GENERATION_PROMPT = """
# Given the following problem description, write a function with test cases to test the solution program for this problem.
# The test cases should follow equivalent partitioning and boundary value analysis.

# For example:
# Input: 
# def sum(a: int, b: int):
#     \"\"\" Return the sume of two integers 
#     >>> sum(1, 2)
#     3
#     >>> sum(0, 0)
#     0
#     \"\"\"
# Output:
# First, the input category can be divided into several categories:
# 1. Positive integers: Both a and b are positive.
# 2. Negative integers: Both a and b are negative.
# 3. Zero values: Either a, b, or both are zero.
# 4. Mixed signs: One is positive, and the other is negative.
# 5. Large integers: a and/or b are very large (to check overflow behavior, though Python handles big integers).

# To generate a test case for each category, the test case should be:
# ```python
# def check(candidate):``
#     # positive integers
#     assert candidate(1, 2) == 3
#     # negative integers
#     assert candidate(-1, -2) == -3
#     # zero values
#     assert candidate(0, 5) == 5
#     # mixed signs
#     assert candidate(7, -2) == 5
#     # large integers
#     assert candidate(1_000_000, 2_000_000) == 3_000_000
# ```

# Make sure the test cases are valid and do not include any potentially ambiguous cases.
# Make sure the input satisfy the input constraints and do not include any invalid inputs.
# Make sure to consider the precision issues when comparing floating-point numbers.
# Follow the above example, first output the equivalent calsses and then generate one test case for each class in a Markdown code block.

# Input: {problem}
# Output:
# """


TEST_GENERATION_PROMPT = """
Given the following problem description, write a function with test cases to test the solution program for this problem.

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

def generate_test_cases(llm: BaseChatModel, problem: dict) -> str:
    prompt = PromptTemplate.from_template(TEST_GENERATION_PROMPT)
    chain = prompt | llm | PythonOutputParser()

    return chain.invoke({
        "problem": problem["prompt"]
    })


def evaluate_solution(problem: dict, completion: str) -> bool:
    """
    Evaluates the correctness of a predicted solution to a problem prompt.

    Check the human eval dataset on how the programs are executed.
    Then, execute these programs and compare the output with the expected output.
    To see the percentage of incorrect functions that can be detected by the 
    test cases  
    """
    print(problem["prompt"])
    examples = PromptExampleParser().parse(problem["prompt"])
    print(examples)
    test_str = construct_test(examples)
    problem["test"] = test_str

    return check_correctness(problem, completion, 3.0)

def construct_test(examples: list[dict]) -> str:
    test_str = "def check(candidate):\n"
    for example in examples:
        test_str += f"    assert {example['input']} == {example['output']}\n"

    if len(examples) == 0:
        test_str += "    pass\n"

    return test_str


if __name__ == "__main__":
    samples = []
    with open("samples.jsonl_results.jsonl", "r") as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)

    problems = read_problems()

    sample = samples[1]
    problem = problems[sample["task_id"]]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.01)

    test_str = generate_test_cases(llm, problem)

    print(problem["test"])
    print(problem["prompt"])
    print(sample["completion"])
    print(test_str)
    problem["test"] = test_str
    result = check_correctness(problem, sample["completion"], 3.0)
    print(result)
    # results = []
    # for sample in samples[47:48]:
    #     # if sample["passed"]:
    #     #     continue

    #     problem = problems[sample["task_id"]]
    #     print(problem["test"])
    #     completion = sample["completion"]
    #     passed = evaluate_solution(problem, completion)
    #     results.append(passed)

    # passed = [result["passed"] for result in results]
    # gt_passed = [result["passed"] for result in samples]
    # print(results)
    # for i in range(len(results)):
    #     if gt_passed[i] and not passed[i]:
    #         print(f"Unexpected error {i}")
    #         break
    # print(f"Passed: {sum(passed)}/{len(results)}")
