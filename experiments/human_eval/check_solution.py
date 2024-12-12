from parsers import PromptExampleParser

def evaluate_solution(prompt: str, predicted_solution: str) -> bool:
    """
    Evaluates the correctness of a predicted solution to a problem prompt.
    """
    examples = PromptExampleParser().parse(prompt)
    ## TODO: Check the human eval dataset on how the programs are executed.
    # Then, execute these programs and compare the output with the expected output.
    # To see the percentage of incorrect functions that can be detected by the 
    # test cases 