from sherpa_ai.actions.base import BaseAction
from langchain_core.language_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from llm_coder.agent_coder.prompts import CODE_GENERATION_PROMPT, TEST_GENERATION_PROMPT
from llm_coder.agent_coder.parsers import PythonProgramParser, PythonTestParser
from llm_coder.execution import check_correctness
from loguru import logger


class GenerateProgram(BaseAction):
    name: str = "generate_program"
    args: dict = {}
    usage: str = "Generate a program for the programming problem."

    llm: BaseChatModel
    parser: PythonProgramParser = PythonProgramParser()

    def execute(self) -> str:
        """
        Generate a program for the programming problem.

        Returns:
            str: The generated program
        """
        logger.info("Solving problem with agent coder")
        prompt_template = ChatPromptTemplate(
            messages=[
                ("system", "You are a software programmer."),
                ("user", CODE_GENERATION_PROMPT)
            ]
        )

        problem_description = self.belief.get("problem")["prompt"]
        chain = prompt_template | self.llm | self.parser
        result = chain.invoke({"problem_description": problem_description})
        num_llm_calls = self.belief.get("num_llm_calls", 0)
        self.belief.set("num_llm_calls", num_llm_calls + 1)

        generated_programs = self.belief.get("generated_programs", [])
        generated_programs.append(result)
        self.belief.set("generated_programs", generated_programs)

        return result


class GenerateTest(BaseAction):
    name: str = "generate_test"
    args: dict = {}
    usage: str = "Generate test cases for the programming problem."

    llm: BaseChatModel
    parser: PythonTestParser = PythonTestParser()

    def execute(self) -> str:
        """
        Generate test cases for the programming problem.

        Returns:
            str: The generated test cases
        """
        logger.info("Generating test cases with agent coder")
        prompt_template = ChatPromptTemplate(
            messages=[
                ("system", "You are a code developer assistant."),
                ("user", TEST_GENERATION_PROMPT)
            ]
        )

        problem_description = self.belief.get("problem")["prompt"]
        chain = prompt_template | self.llm | self.parser
        result = chain.invoke({"problem_description": problem_description})
        num_llm_calls = self.belief.get("num_llm_calls", 0)
        self.belief.set("num_llm_calls", num_llm_calls + 1)
        logger.info(result)

        generated_tests = self.belief.get("generated_tests", [])
        generated_tests.append(result)
        self.belief.set("generated_tests", generated_tests)

        return result


class EvaluateProgram(BaseAction):
    name: str = "evaluate_program"
    args: dict = {}
    usage: str = "Evaluate the generated program."

    total_count: int = 5
    current_count: int = 0

    def execute(self) -> str:
        logger.info("Evaluating the generated program.")
        self.current_count += 1

        problem = self.belief.get("problem")

        problem = problem.copy()
        solutions= self.belief.get("generated_programs")
        test_cases = self.belief.get("generated_tests")

        for solution in solutions:
            passed = 0
            for test_case in test_cases:
                test_case = self.create_test(test_case, problem["entry_point"])
                problem["test"] = test_case
                evaluation_result = check_correctness(problem, solution, 3.0)
                
                if evaluation_result["passed"]:
                    passed += 1

            if passed >= 3: # use the same number as in the AgentCoder
                self.belief.set("evaluation_passed", passed)
                if passed:
                    self.belief.set("generated_solution", solution)
                return True

        # If the current count is greater than or equal to the total count,
        # then stop
        if self.current_count >= self.total_count:
            passed = True
            self.belief.set("evaluation_passed", passed)
            self.belief.set("generated_solution", solutions[-1])

        return True

    def create_test(self, test_str, entry_point):
        test_str = test_str.strip().replace(entry_point, "candidate")
        test_cases = [f"    {test_case}" for test_case in test_str.split("\n")]

        return f"def check(candidate):\n{'\n'.join(test_cases)}\n"
    
    def is_passed(self):
        return self.belief.get("evaluation_passed", False)
