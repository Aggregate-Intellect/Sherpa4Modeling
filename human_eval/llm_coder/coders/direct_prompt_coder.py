from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from llm_coder.parsers import PythonOutputParser
from llm_coder.coders.base import BaseCoder, CoderResult

PROMPT_TEMPLATE = """Complete the following code. Use ```python to put the completed Python code, including the necessary imports, in markdown quotes:\n{problem}"""


class DirectPromptCoder(BaseCoder):
    llm: BaseChatModel = None
    parser: PythonOutputParser = PythonOutputParser()

    def solve_problem(self, problem: dict):
        problem = problem["prompt"]
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | self.llm | self.parser
        result = None
        while result is None:
            result = chain.invoke({
                "problem": problem
            })
        return CoderResult(result=result, num_llm_calls=1)
