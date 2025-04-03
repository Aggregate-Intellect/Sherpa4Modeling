from sherpa_ai.actions.base import BaseAction
from langchain_core.language_models import BaseChatModel
from llm_coder.parsers import PythonOutputParser, TestCaseParser
from langchain.prompts import PromptTemplate
from llm_coder.test_based_sm.prompts import ITERATIVE_PROMPT, FIRST_GENERATION_PROMPT, TEST_GENERATION_PROMPT
from llm_coder.execution import check_correctness
from loguru import logger


class GenerateSolution(BaseAction):
    name: str = "generate_solution"
    args: dict = {}
    usage: str = "Generate a solution for the programming problem."

    # Whether to consider the previous solution when generating a new solution
    iterative: bool = True

    llm: BaseChatModel
    parser: PythonOutputParser = PythonOutputParser()

    def execute(self) -> str:
        """
        Generate a solution for the programming problem.

        When called the first time, the action will generate a solution from scratch.
        If the action is called again, it will generate a solution based on the existing solution and test cases.

        Returns:
            str: The generated solution
        """
        logger.info("Generating solution for the programming problem.")
        problem = self.belief.get("problem")["prompt"]
        if self.belief.get("generated_solution", None) is not None:
            chain = PromptTemplate.from_template(
                ITERATIVE_PROMPT) | self.llm | self.parser
            input_data = {
                "problem": problem,
                "existing_solution": self.belief.get("generated_solution"),
                "test_cases": self.belief.get("test_cases")
            }
        else:
            chain = PromptTemplate.from_template(
                FIRST_GENERATION_PROMPT) | self.llm | self.parser
            input_data = {
                "problem": problem
            }

        result = None

        while result is None:
            result = chain.invoke(input_data)
            num_llm_calls = self.belief.get("num_llm_calls", 0)
            self.belief.set("num_llm_calls", num_llm_calls + 1)
        if self.iterative:
            self.belief.set("generated_solution", result)

        return result


class GenerateTestCases(BaseAction):
    name: str = "generate_test_cases"
    args: dict = {}
    usage: str = "Generate test cases for the programming problem."

    llm: BaseChatModel
    parser: PythonOutputParser = PythonOutputParser()
    test_cases_parser: TestCaseParser = TestCaseParser()

    def execute(self) -> str:
        """
        Generate test cases for the programming problem.

        Returns:
            str: The generated test cases
        """
        logger.info("Generating test cases for the programming problem.")
        problem = self.belief.get("problem")["prompt"]
        chain = PromptTemplate.from_template(
            TEST_GENERATION_PROMPT) | self.llm | self.parser | self.test_cases_parser

        result = None
        while result is None:
            result = chain.invoke({
                "problem": problem
            })
            num_llm_calls = self.belief.get("num_llm_calls", 0)
            self.belief.set("num_llm_calls", num_llm_calls + 1)
        self.belief.set("test_cases", result)

        return result


class EvaluateSolution(BaseAction):
    name: str = "evaluate_solution"
    args: dict = {}
    usage: str = "Evaluate the correctness of the solution."
    total_count: int = 5
    current_count: int = 0

    def execute(self) -> bool:
        """
        Evaluate the correctness of the solution.

        Returns:
            bool: True if the solution is correct, False otherwise
        """
        logger.info("Evaluating the correctness of the solution.")
        self.current_count += 1
        problem = self.belief.get("problem")
        problem = problem.copy()

        # Support both iterative setting (stored as "generated solution")
        # and non-iterative setting (stored as "generate_solution")
        solution = self.belief.get("generated_solution", self.belief.get("generate_solution"))
        test_cases = self.belief.get("test_cases")

        passed_test_cases = 0
        for test_case in test_cases:
            problem["test"] = self.create_test(test_case)
            logger.info(self.create_test(test_case))
            evaluation_result = check_correctness(problem, solution, 3.0)
            if evaluation_result["passed"]:
                passed_test_cases += 1

        if passed_test_cases == len(test_cases):
            self.belief.set("solution_correct", True)
            self.belief.set("generated_solution", solution)
            return True

        generated_candidates = self.belief.get("generated_candidates", [])
        generated_candidates.append({
            "solution": solution,
            "test_case_passed_ratio": passed_test_cases / len(test_cases)
        })
        self.belief.set("generated_candidates", generated_candidates)

        if self.current_count < self.total_count:
            self.belief.set("solution_correct", False)
            return False

        self.current_count = 0
        best_solution = max(
            generated_candidates, key=lambda x: x["test_case_passed_ratio"])["solution"]
        self.belief.set("generated_solution", best_solution)
        self.belief.set("solution_correct", True)
        return True

    def create_test(self, test_case):
        return f"def check(candidate):\n    {test_case}\n"


class CheckCorrectness(BaseAction):
    name: str = "check_correctness"
    args: dict = {}
    usage: str = "Check the correctness of the solution."

    def execute(self) -> bool:
        """
        Check the correctness of the solution.

        Returns:
            bool: True if the solution is correct, False otherwise
        """

        return self.belief.get("solution_correct", False)
